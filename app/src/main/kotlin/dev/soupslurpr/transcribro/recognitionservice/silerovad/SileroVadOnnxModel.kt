package dev.soupslurpr.transcribro.recognitionservice.silerovad

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import dev.soupslurpr.transcribro.recognitionservice.OnnxSchemaValidator

class SileroVadOnnxModel(modelBytes: ByteArray) {
    private val session: OrtSession

    // Silero-VAD v5+ (the upstream `silero_vad.onnx` we now ship) packs the LSTM state
    // into a single tensor `state[2, batch, 128]`. v3-era models used separate
    // h[2,batch,64] / c[2,batch,64] tensors and a different input list — see git history
    // if you ever need to bring an old export back.
    private lateinit var state: Array<Array<FloatArray>>

    // v5+ also requires every call to be a fixed-size frame (512 samples @ 16 kHz, 256
    // samples @ 8 kHz) preceded by a CONTEXT_SIZE-sample carry-over from the previous
    // frame. We maintain that rolling context here so [call] can pretend the API still
    // takes raw samples. Reset along with [state].
    private lateinit var context: Array<FloatArray>

    private var lastSr = 0
    private var lastBatchSize = 0

    init {
        val env: OrtEnvironment = OrtEnvironment.getEnvironment()
        val opts: OrtSession.SessionOptions = OrtSession.SessionOptions().apply {
            setInterOpNumThreads(1)
            setIntraOpNumThreads(1)
            addCPU(true)
        }
        session = env.createSession(modelBytes, opts)
        OnnxSchemaValidator.verify(
            modelName = "silero_vad",
            session = session,
            expectedInputs = setOf("input", "state", "sr"),
            expectedOutputCount = 2,
        )
        resetStates()
    }

    fun resetStates() {
        state = Array(2) { Array(1) { FloatArray(128) } }
        // Initialised lazily once we know the sample rate (and therefore CONTEXT_SIZE).
        context = arrayOf(FloatArray(0))
        lastSr = 0
        lastBatchSize = 0
    }

    @Throws(OrtException::class)
    fun close() {
        session.close()
    }

    /**
     * Define inner class ValidationResult
     */
    class ValidationResult // Constructor
        (val x: Array<FloatArray?>, val sr: Int)

    /**
     * Function to validate input data
     */
    private fun validateInput(x: Array<FloatArray?>, sr: Int): ValidationResult {
        // Process the input data with dimension 1
        var x = x
        var sr = sr
        if (x.size == 1) {
            x = arrayOf(x[0])
        }
        // Throw an exception when the input data dimension is greater than 2
        require(x.size <= 2) { "Incorrect audio data dimension: " + x[0]!!.size }

        // Process the input data when the sample rate is not equal to 16000 and is a multiple of 16000
        if (sr != 16000 && (sr % 16000 == 0)) {
            val step = sr / 16000
            val reducedX = arrayOfNulls<FloatArray>(x.size)

            for (i in x.indices) {
                val current = x[i]
                val newArr = FloatArray((current!!.size + step - 1) / step)

                var j = 0
                var index = 0
                while (j < current.size) {
                    newArr[index] = current[j]
                    j += step
                    index++
                }

                reducedX[i] = newArr
            }

            x = reducedX
            sr = 16000
        }

        // If the sample rate is not in the list of supported sample rates, throw an exception
        require(SAMPLE_RATES.contains(sr)) { "Only supports sample rates $SAMPLE_RATES (or multiples of 16000)" }

        // v5+ requires exactly FRAME_SIZE samples per call; reject anything else here so
        // the failure mode is a clear `IllegalArgumentException` instead of a buried
        // `OrtException` from the LSTM.
        val expected = frameSize(sr)
        require(x[0]!!.size == expected) {
            "silero v5+ requires exactly $expected samples per call at $sr Hz; got ${x[0]!!.size}"
        }

        return ValidationResult(x, sr)
    }

    /**
     * Method to call the ONNX model
     */
    @Throws(OrtException::class)
    fun call(x: Array<FloatArray?>, sr: Int): FloatArray {
        var x = x
        var sr = sr
        val result = validateInput(x, sr)
        x = result.x
        sr = result.sr

        val batchSize = x.size
        val ctxSize = contextSize(sr)
        val frameLen = frameSize(sr)

        if (lastBatchSize == 0 || lastSr != sr || lastBatchSize != batchSize) {
            resetStates()
        }
        // First call after reset (or sample-rate change): start with zero context.
        if (context[0].size != ctxSize) {
            context = arrayOf(FloatArray(ctxSize))
        }

        // Build [batch, ctxSize + frameLen] = previous-context ++ this-frame
        val frame = x[0]!!
        val combined = Array(1) { FloatArray(ctxSize + frameLen) }
        System.arraycopy(context[0], 0, combined[0], 0, ctxSize)
        System.arraycopy(frame, 0, combined[0], ctxSize, frameLen)

        val env: OrtEnvironment = OrtEnvironment.getEnvironment()

        var inputTensor: OnnxTensor? = null
        var stateTensor: OnnxTensor? = null
        var srTensor: OnnxTensor? = null
        var ortOutputs: OrtSession.Result? = null

        try {
            inputTensor = OnnxTensor.createTensor(env, combined)
            stateTensor = OnnxTensor.createTensor(env, state)
            // v5+ silero expects `sr` as a 0-d (scalar) int64 tensor, not [1].
            srTensor = OnnxTensor.createTensor(
                env,
                java.nio.LongBuffer.wrap(longArrayOf(sr.toLong())),
                longArrayOf(),
            )

            val inputs: MutableMap<String, OnnxTensor?> = HashMap()
            inputs["input"] = inputTensor
            inputs["state"] = stateTensor
            inputs["sr"] = srTensor

            ortOutputs = session.run(inputs)
            @Suppress("UNCHECKED_CAST")
            val output = ortOutputs.get(0).value as Array<FloatArray>
            @Suppress("UNCHECKED_CAST")
            state = ortOutputs.get(1).value as Array<Array<FloatArray>>

            // Save the trailing ctxSize samples of this frame for the next call.
            val newContext = FloatArray(ctxSize)
            System.arraycopy(frame, frameLen - ctxSize, newContext, 0, ctxSize)
            context = arrayOf(newContext)

            lastSr = sr
            lastBatchSize = batchSize
            return output[0]
        } finally {
            inputTensor?.close()
            stateTensor?.close()
            srTensor?.close()
            ortOutputs?.close()
        }
    }

    companion object {
        private val SAMPLE_RATES: List<Int> = mutableListOf(8000, 16000)

        /** Number of audio samples per inference call. v5+ silero is fixed at this size. */
        fun frameSize(sr: Int): Int = if (sr == 16000) 512 else 256

        /** Number of trailing samples of each frame that get prepended to the next call. */
        fun contextSize(sr: Int): Int = if (sr == 16000) 64 else 32
    }
}