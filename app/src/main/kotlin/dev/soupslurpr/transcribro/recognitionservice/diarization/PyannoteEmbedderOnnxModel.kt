package dev.soupslurpr.transcribro.recognitionservice.diarization

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import dev.soupslurpr.transcribro.recognitionservice.OnnxSchemaValidator
import kotlin.math.sqrt

/**
 * Wraps the WeSpeaker speaker-embedding ONNX model used by pyannote v3.
 *
 * Expected export: sherpa-onnx's `wespeaker_en_voxceleb_resnet34_LM.onnx`
 *   https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
 *
 * Interface (per the sherpa-onnx export):
 *   - input  name "feats"  shape [1, T, 80]    (kaldi-style 80-bin log-mel-fbank)
 *   - output name "embs"   shape [1, 256]      (NOT L2-normalized — caller normalizes)
 *
 * Sample-amplitude convention: kaldi-native-fbank with `normalize_samples = false`,
 * i.e. raw int16 amplitude range. [KaldiFbank] is configured to match.
 *
 * Recommended audio chunk: ≥1.5 s of speech for stable embeddings; very short chunks
 * (<0.5 s) produce noisy embeddings that drift in cosine space and can confuse the
 * speaker registry. Filter those out at the call site.
 */
class PyannoteEmbedderOnnxModel(
    modelBytes: ByteArray,
    private val fbank: KaldiFbank = KaldiFbank(),
) {
    private val session: OrtSession

    init {
        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            setInterOpNumThreads(1)
            setIntraOpNumThreads(2)
            addCPU(true)
        }
        session = env.createSession(modelBytes, opts)
        OnnxSchemaValidator.verify(
            modelName = "pyannote-embedder",
            session = session,
            expectedInputs = setOf(INPUT_NAME),
            expectedOutputCount = 1,
        )
    }

    /**
     * Compute a 256-dim L2-normalized speaker embedding from a chunk of audio.
     *
     * @param audio float-valued samples in **int16 amplitude range** (≈[-32768, 32767]),
     *   16kHz mono. Use [shortsToFbankAudio] to convert from `ShortArray`.
     * @return embedding of length [EMBEDDING_DIM], L2-normalized so that
     *   `dot(a, b)` is the cosine similarity.
     */
    fun embed(audio: FloatArray): FloatArray {
        require(audio.size >= MIN_AUDIO_SAMPLES) {
            "embedder needs at least $MIN_AUDIO_SAMPLES samples (≈${MIN_AUDIO_SAMPLES / 16000f}s); got ${audio.size}"
        }
        val feats = fbank.compute(audio)
        require(feats.isNotEmpty()) { "fbank produced 0 frames for ${audio.size}-sample input" }

        // Cepstral mean subtraction — match wespeaker's reference extraction pipeline
        // (`feats = feats - feats.mean(axis=0)` in cli/extractor.py). Without this, the
        // embedding encodes channel/volume/room-acoustic information common to all
        // voices in the same recording, collapsing cross-speaker cosine distances. We
        // saw distinct voices clustering at ~0.31 before this was added; with CMVN they
        // separate to the wespeaker-published ~0.6+ range.
        val numFrames = feats.size
        val numBins = feats[0].size
        val featMean = FloatArray(numBins)
        for (frame in feats) {
            for (b in 0 until numBins) featMean[b] += frame[b]
        }
        for (b in 0 until numBins) featMean[b] /= numFrames
        for (frame in feats) {
            for (b in 0 until numBins) frame[b] -= featMean[b]
        }

        // ONNX input: [1, T, 80]
        val input = arrayOf(feats)

        val env = OrtEnvironment.getEnvironment()
        var inputTensor: OnnxTensor? = null
        var ortOutputs: OrtSession.Result? = null
        try {
            inputTensor = OnnxTensor.createTensor(env, input)
            ortOutputs = session.run(mapOf(INPUT_NAME to inputTensor))
            @Suppress("UNCHECKED_CAST")
            val embs = ortOutputs.get(0).value as Array<FloatArray>
            val raw = embs[0]
            require(raw.size == EMBEDDING_DIM) {
                "expected $EMBEDDING_DIM-dim embedding, got ${raw.size}"
            }
            return l2Normalize(raw)
        } finally {
            inputTensor?.close()
            ortOutputs?.close()
        }
    }

    fun close() {
        session.close()
    }

    companion object {
        const val SAMPLE_RATE = 16000
        const val EMBEDDING_DIM = 256

        /** Below this much audio, embeddings are too noisy to cluster reliably (~0.5 s). */
        const val MIN_AUDIO_SAMPLES = SAMPLE_RATE / 2

        private const val INPUT_NAME = "feats"

        /**
         * Convert `ShortArray` (PCM-16) to the float buffer the embedder expects.
         * Note we do NOT divide by 32768 — kaldi-native-fbank with normalize_samples=false
         * expects raw int16 amplitude range, and that's what wespeaker was trained on.
         */
        fun shortsToFbankAudio(samples: ShortArray): FloatArray =
            FloatArray(samples.size) { samples[it].toFloat() }

        private fun l2Normalize(v: FloatArray): FloatArray {
            var sumSq = 0.0
            for (x in v) sumSq += x * x
            val norm = sqrt(sumSq).toFloat()
            if (norm == 0f) return v.copyOf()
            return FloatArray(v.size) { v[it] / norm }
        }
    }
}
