package com.whispercpp.rnnoise

import android.util.Log
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

private const val TAG = "LibRnnoise"

/**
 * Streaming wrapper around xiph/rnnoise.
 *
 * RNNoise wants 480-sample frames of 48 kHz audio in int16 amplitude range (as
 * float). Our pipeline runs at 16 kHz, so this class internally:
 *
 *   1. Resamples 16 kHz → 48 kHz on input (polyphase upsampler, L=3, 33-tap FIR LPF).
 *   2. Buffers to 480-sample (48 kHz) frames.
 *   3. Calls `nativeProcessFrame` on each full frame.
 *   4. Resamples 48 kHz → 16 kHz on output (polyphase downsampler, M=3, same FIR LPF).
 *   5. Returns whatever 16 kHz denoised samples are now ready.
 *
 * Latency: one 480-sample 48 kHz frame = 10 ms wall-clock. The output stream lags
 * the input by ~10 ms plus the filter's group delay (~0.3 ms).
 *
 * Not thread-safe — `process` keeps internal buffers / resampler state. Caller
 * serializes. One instance per audio session.
 */
class Denoiser private constructor(private var ptr: Long) {

    private val upsampler = Upsampler3x()
    private val downsampler = Downsampler3x()

    // Accumulator for 48 kHz samples while we wait for a full 480-sample frame.
    private val frameAccumulator = FloatArray(FRAME_SIZE)
    private var frameAccumulatorFill = 0

    // Output queue of denoised 16 kHz samples. Grows as 480-sample 48 kHz frames are
    // processed and shrinks as the caller consumes them. We back it with a plain
    // ShortArray that grows via copy — typical call patterns produce <= a few hundred
    // samples per [process] call, so the cost is negligible.
    private var pendingOutput: ShortArray = ShortArray(0)
    private var pendingOutputSize: Int = 0

    /**
     * Feed [input] (16 kHz mono int16) and return any denoised samples that became
     * available. Output sample count may be more or less than [input] size — it lags
     * by one 480-sample 48 kHz frame (10 ms) on session start, and may emit a partial
     * frame's worth on subsequent calls depending on buffering.
     */
    fun process(input: ShortArray): ShortArray {
        require(ptr != 0L) { "Denoiser used after close()" }
        if (input.isEmpty()) return drainPending()

        // 16 kHz → 48 kHz. Each input sample produces exactly 3 output samples.
        val upsampled = upsampler.process(input)

        var consumed = 0
        while (consumed < upsampled.size) {
            val want = FRAME_SIZE - frameAccumulatorFill
            val take = minOf(want, upsampled.size - consumed)
            System.arraycopy(upsampled, consumed, frameAccumulator, frameAccumulatorFill, take)
            frameAccumulatorFill += take
            consumed += take

            if (frameAccumulatorFill == FRAME_SIZE) {
                // Full 10 ms frame ready. Denoise in-place, then downsample back to 16 kHz.
                nativeProcessFrame(ptr, frameAccumulator)
                val down = downsampler.process(frameAccumulator)
                appendPending(down)
                frameAccumulatorFill = 0
            }
        }

        return drainPending()
    }

    fun close() {
        if (ptr != 0L) {
            nativeDestroy(ptr)
            ptr = 0L
        }
    }

    private fun appendPending(samples: ShortArray) {
        val needed = pendingOutputSize + samples.size
        if (needed > pendingOutput.size) {
            // Grow at least 2x to amortize copies, but cap initial allocation modestly.
            val newCap = maxOf(needed, pendingOutput.size * 2, 480)
            pendingOutput = pendingOutput.copyOf(newCap)
        }
        System.arraycopy(samples, 0, pendingOutput, pendingOutputSize, samples.size)
        pendingOutputSize += samples.size
    }

    private fun drainPending(): ShortArray {
        if (pendingOutputSize == 0) return EMPTY_SHORT
        val out = pendingOutput.copyOf(pendingOutputSize)
        pendingOutputSize = 0
        return out
    }

    companion object {
        const val SAMPLE_RATE_HZ: Int = 16000
        const val FRAME_SIZE: Int = 480  // RNNoise's native frame size at 48 kHz (10 ms).

        private val EMPTY_SHORT = ShortArray(0)

        init {
            System.loadLibrary("rnnoise")
        }

        fun create(): Denoiser {
            val ptr = nativeCreate()
            check(ptr != 0L) { "rnnoise_create returned null" }
            Log.i(TAG, "rnnoise denoiser created (frameSize=${nativeFrameSize()})")
            return Denoiser(ptr)
        }

        @JvmStatic external fun nativeCreate(): Long
        @JvmStatic external fun nativeDestroy(ptr: Long)
        @JvmStatic external fun nativeProcessFrame(ptr: Long, frame: FloatArray): Float
        @JvmStatic external fun nativeFrameSize(): Int
    }
}

/**
 * 33-tap windowed-sinc low-pass FIR designed for the 3× resampler. Cutoff is 7 kHz
 * at the 48 kHz inner rate (0.146 normalized), with a Hamming window for moderate
 * sidelobe suppression. 33 taps is a deliberately modest choice: ample for clean
 * speech-band reproduction, cheap to compute, group delay is 16/48000 ≈ 0.33 ms.
 *
 * The same coefficients drive both directions — for upsampling, the L=3 polyphase
 * subfilters consume every third coefficient; for downsampling, all coefficients
 * are convolved over the 48 kHz stream before decimation.
 */
private object ResampleFilter {
    const val NUM_TAPS = 33
    const val L = 3                  // interpolation / decimation factor
    val TAPS: FloatArray = designLowpass(NUM_TAPS, cutoffNormalized = 7000.0 / 48000.0)

    private fun designLowpass(numTaps: Int, cutoffNormalized: Double): FloatArray {
        val coeffs = DoubleArray(numTaps)
        val center = (numTaps - 1) / 2.0
        // 2 * fc * sinc(2*fc*(n-center)) * hamming(n)
        var sum = 0.0
        for (n in 0 until numTaps) {
            val k = n - center
            val sinc = if (k == 0.0) {
                2.0 * cutoffNormalized
            } else {
                val arg = 2.0 * PI * cutoffNormalized * k
                sin(arg) / (PI * k)
            }
            val hamming = 0.54 - 0.46 * cos(2.0 * PI * n / (numTaps - 1))
            coeffs[n] = sinc * hamming
            sum += coeffs[n]
        }
        // Normalize DC gain to exactly 1.0 (sinc + window has tiny numeric drift).
        val out = FloatArray(numTaps)
        for (n in 0 until numTaps) out[n] = (coeffs[n] / sum).toFloat()
        return out
    }
}

/**
 * Polyphase upsampler: 16 kHz → 48 kHz with the [ResampleFilter] coefficients,
 * compensated for the L=3 zero-stuffing energy loss (gain ×3).
 *
 * State: a ring of the last ⌈numTaps/L⌉ input samples — the polyphase decomposition
 * means each output uses only that many input samples (instead of all 33 of the
 * underlying filter operating on the zero-stuffed stream).
 */
private class Upsampler3x {
    private val L = ResampleFilter.L
    private val numTaps = ResampleFilter.NUM_TAPS
    // Each of the L subfilters has ⌈numTaps/L⌉ coefficients (11 here for 33/3).
    private val subTaps = (numTaps + L - 1) / L
    private val subfilters: Array<FloatArray> = Array(L) { phase ->
        FloatArray(subTaps) { k ->
            val idx = k * L + phase
            // Compensate for energy loss from zero-stuffing.
            if (idx < numTaps) ResampleFilter.TAPS[idx] * L else 0f
        }
    }
    private val history = FloatArray(subTaps)

    fun process(input: ShortArray): FloatArray {
        val out = FloatArray(input.size * L)
        for (i in input.indices) {
            // Shift history (oldest -> dropped). Index 0 = newest.
            for (h in subTaps - 1 downTo 1) history[h] = history[h - 1]
            history[0] = input[i].toFloat()
            // Emit L output samples — one per polyphase subfilter.
            for (phase in 0 until L) {
                val taps = subfilters[phase]
                var acc = 0f
                for (k in 0 until subTaps) acc += taps[k] * history[k]
                out[i * L + phase] = acc
            }
        }
        return out
    }
}

/**
 * Polyphase downsampler: 48 kHz → 16 kHz. Convolves [ResampleFilter] across the
 * 48 kHz stream and keeps every M-th output. Polyphase rewrite: only compute the
 * sums that will be kept (one per L=3 input samples), so the cost is 11 MACs per
 * 16 kHz output sample instead of 33.
 */
private class Downsampler3x {
    private val M = ResampleFilter.L  // decimation factor matches L (3)
    private val numTaps = ResampleFilter.NUM_TAPS
    private val history = FloatArray(numTaps)
    private var phase = 0  // tracks position within each M-sample input group

    fun process(input: FloatArray): ShortArray {
        // We emit one output per M input samples on average; size output upper-bound,
        // then trim.
        val maxOut = (input.size + M - 1) / M + 1
        val out = ShortArray(maxOut)
        var outCount = 0
        for (i in input.indices) {
            for (h in numTaps - 1 downTo 1) history[h] = history[h - 1]
            history[0] = input[i]
            if (phase == 0) {
                var acc = 0f
                val taps = ResampleFilter.TAPS
                for (k in 0 until numTaps) acc += taps[k] * history[k]
                out[outCount++] = clipToShort(acc)
            }
            phase = (phase + 1) % M
        }
        return out.copyOf(outCount)
    }

    private fun clipToShort(v: Float): Short {
        val rounded = v.toInt()
        return when {
            rounded > Short.MAX_VALUE.toInt() -> Short.MAX_VALUE
            rounded < Short.MIN_VALUE.toInt() -> Short.MIN_VALUE
            else -> rounded.toShort()
        }
    }
}
