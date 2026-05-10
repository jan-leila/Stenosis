package dev.soupslurpr.transcribro.recognitionservice.diarization

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import dev.soupslurpr.transcribro.recognitionservice.OnnxSchemaValidator

/**
 * Wraps the pyannote/segmentation-3.0 ONNX model.
 *
 * Expected export: the sherpa-onnx export at
 *   https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx
 *
 * That export has:
 *   - input  name "x"  shape [batch, channel, samples] = [1, 1, 160000] (10s @ 16kHz mono)
 *   - output name "y"  shape [batch, frames, classes]  = [1, 589, 7]    (log-probs, powerset)
 *
 * The 7 output classes are the powerset of {speaker0, speaker1, speaker2} restricted to
 * at-most-2 simultaneous speakers:
 *   0: silent     1: {0}     2: {1}     3: {2}     4: {0,1}     5: {0,2}     6: {1,2}
 *
 * Note: the pengzhendong/pyannote-onnx export uses different I/O names ("input"…) — if you
 * ever swap exports, [INPUT_NAME] needs to change to match.
 *
 * "Local speakers" here are window-scoped (0/1/2 within this 10s window). The same physical
 * speaker may receive a different local index in a different window. To assign stable global
 * IDs across windows, pair this with [PyannoteEmbedderOnnxModel] + a speaker registry.
 */
class PyannoteSegmenterOnnxModel(modelBytes: ByteArray) {
    private val session: OrtSession

    init {
        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            // Segmentation is small (~6MB int8) — modest IntraOp parallelism is plenty,
            // and we don't run multiple ops in parallel here.
            setInterOpNumThreads(1)
            setIntraOpNumThreads(2)
            addCPU(true)
        }
        session = env.createSession(modelBytes, opts)
        OnnxSchemaValidator.verify(
            modelName = "pyannote-segmentation",
            session = session,
            expectedInputs = setOf(INPUT_NAME),
            expectedOutputCount = 1,
        )
    }

    /**
     * Run segmentation on a single 10-second window.
     *
     * @param audio FloatArray of exactly [WINDOW_SAMPLES] samples, 16kHz mono, normalized to [-1, 1]
     * @return per-frame boolean activity matrix (shape [NUM_FRAMES][NUM_LOCAL_SPEAKERS])
     */
    fun segment(audio: FloatArray): SegmentationResult {
        require(audio.size == WINDOW_SAMPLES) {
            "Pyannote segmentation expects exactly $WINDOW_SAMPLES samples (10s @ 16kHz); got ${audio.size}"
        }
        val env = OrtEnvironment.getEnvironment()
        // Reshape to [1, 1, samples]
        val input = arrayOf(arrayOf(audio))

        var inputTensor: OnnxTensor? = null
        var ortOutputs: OrtSession.Result? = null
        try {
            inputTensor = OnnxTensor.createTensor(env, input)
            ortOutputs = session.run(mapOf(INPUT_NAME to inputTensor))
            @Suppress("UNCHECKED_CAST")
            val raw = ortOutputs.get(0).value as Array<Array<FloatArray>>
            return decodePowerset(raw[0])
        } finally {
            inputTensor?.close()
            ortOutputs?.close()
        }
    }

    private fun decodePowerset(frames: Array<FloatArray>): SegmentationResult {
        val activity = Array(frames.size) { BooleanArray(NUM_LOCAL_SPEAKERS) }
        val classHist = IntArray(POWERSET_CLASSES.size)
        for (i in frames.indices) {
            val classIdx = argmax(frames[i])
            classHist[classIdx]++
            for (sp in POWERSET_CLASSES[classIdx]) {
                activity[i][sp] = true
            }
        }
        return SegmentationResult(activity, classHist)
    }

    private fun argmax(arr: FloatArray): Int {
        var maxIdx = 0
        var maxVal = arr[0]
        for (i in 1 until arr.size) {
            if (arr[i] > maxVal) {
                maxVal = arr[i]
                maxIdx = i
            }
        }
        return maxIdx
    }

    fun close() {
        session.close()
    }

    companion object {
        const val SAMPLE_RATE = 16000
        const val WINDOW_DURATION_S = 10
        const val WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_DURATION_S        // 160000
        const val NUM_FRAMES = 589
        const val NUM_LOCAL_SPEAKERS = 3
        const val FRAME_DURATION_MS: Double = (WINDOW_DURATION_S * 1000.0) / NUM_FRAMES   // ≈16.978

        private const val INPUT_NAME = "x"

        // Powerset(K=3, max_concurrent=2) → 7 classes.
        private val POWERSET_CLASSES: Array<IntArray> = arrayOf(
            intArrayOf(),
            intArrayOf(0),
            intArrayOf(1),
            intArrayOf(2),
            intArrayOf(0, 1),
            intArrayOf(0, 2),
            intArrayOf(1, 2),
        )
    }
}

/**
 * Per-frame, per-(local)-speaker activity output of [PyannoteSegmenterOnnxModel.segment].
 * `activity[frame][localSpeaker] == true` ⇒ that local speaker is talking at that frame.
 *
 * [classHistogram] counts how many frames each powerset class was the argmax for, indexed
 * by class id (silent, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}). Useful for diagnosing whether
 * the model is producing varied outputs or collapsing into a single class.
 */
data class SegmentationResult(
    val activity: Array<BooleanArray>,
    val classHistogram: IntArray,
) {
    val numFrames: Int get() = activity.size
    val numLocalSpeakers: Int get() = if (activity.isNotEmpty()) activity[0].size else 0

    /** True if any local speaker is active at the given frame. */
    fun isAnySpeechAt(frame: Int): Boolean = activity[frame].any { it }

    /**
     * True if [localSpeaker] is the only active speaker at [frame]. Used by the embedder
     * to extract clean single-speaker audio — wespeaker embeddings are very sensitive to
     * background voice content, and including overlap frames (powerset classes 4/5/6)
     * shifts the embedding toward the mixture, spawning phantom speakers in the registry.
     */
    fun isExclusiveAt(frame: Int, localSpeaker: Int): Boolean {
        val row = activity[frame]
        if (!row[localSpeaker]) return false
        for (other in row.indices) {
            if (other != localSpeaker && row[other]) return false
        }
        return true
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is SegmentationResult) return false
        return activity.contentDeepEquals(other.activity) &&
            classHistogram.contentEquals(other.classHistogram)
    }

    override fun hashCode(): Int =
        31 * activity.contentDeepHashCode() + classHistogram.contentHashCode()
}
