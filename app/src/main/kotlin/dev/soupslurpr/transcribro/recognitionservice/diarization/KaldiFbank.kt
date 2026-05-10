package dev.soupslurpr.transcribro.recognitionservice.diarization

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Compute kaldi-compatible mel-filterbank features.
 *
 * The wespeaker_en_voxceleb_resnet34_LM ONNX export consumes 80-dim fbank features
 * computed by `kaldi-native-fbank`. To produce embeddings whose cosine distances are
 * comparable to what wespeaker was trained on, we mirror kaldi's default fbank
 * pipeline as closely as practical:
 *
 *   - Per-frame: DC offset removal → pre-emphasis → povey window → zero-padded real FFT
 *     → power spectrum → triangular HTK-mel filter bank → log
 *   - Frame extraction with `snip_edges = false`, i.e. frame f starts at
 *     `f*shift + shift/2 - frame_length/2` and reads with reflection padding.
 *   - `normalize_samples = false` — input samples are expected in int16 amplitude range
 *     (i.e. roughly [-32768, 32767]), NOT in [-1, 1]. This matters: scaling here changes
 *     the absolute magnitude of the spectrum (which feeds into `log`) and shifts
 *     embeddings.
 *
 * Settings are baked-in to match wespeaker's training-time fbank. Tweak only if you
 * know what you're doing.
 *
 * Output: `[T][80]` where T is frame count. Returned as `Array<FloatArray>`.
 */
class KaldiFbank(
    private val sampleRate: Int = 16000,
    private val frameLengthMs: Int = 25,
    private val frameShiftMs: Int = 10,
    private val numMelBins: Int = 80,
    private val lowFreqHz: Float = 20f,
    private val highFreqHz: Float = -1f,         // -1 ⇒ Nyquist (sampleRate / 2)
    private val preemphCoeff: Float = 0.97f,
    private val removeDcOffset: Boolean = true,
) {
    private val frameLength: Int = sampleRate * frameLengthMs / 1000              // 400 @ 16k
    private val frameShift: Int = sampleRate * frameShiftMs / 1000                // 160 @ 16k
    private val fftSize: Int = nextPowerOfTwo(frameLength)                        // 512
    private val effectiveHighFreq: Float =
        if (highFreqHz <= 0f) sampleRate / 2f else highFreqHz

    private val window: FloatArray = poveyWindow(frameLength)
    private val melFilters: Array<FloatArray> = buildMelFilters(
        fftSize = fftSize,
        sampleRate = sampleRate,
        numMelBins = numMelBins,
        lowFreq = lowFreqHz,
        highFreq = effectiveHighFreq,
    )

    /** Number of frames produced for `audio.size` input samples (snip_edges = false). */
    fun numFrames(numSamples: Int): Int {
        if (numSamples <= 0) return 0
        // (n + shift/2) / shift, then drop trailing frames whose right edge would overflow.
        var n = (numSamples + frameShift / 2) / frameShift
        while (n > 0) {
            val firstSample = firstSampleOfFrame(n - 1)
            val endSample = firstSample + frameLength
            if (endSample <= numSamples) break
            n--
        }
        return n
    }

    /**
     * Compute the [T, numMelBins] log-mel-filterbank.
     *
     * @param audio float-valued samples in int16 amplitude range (≈[-32768, 32767]).
     */
    fun compute(audio: FloatArray): Array<FloatArray> {
        val n = numFrames(audio.size)
        if (n == 0) return emptyArray()

        val out = Array(n) { FloatArray(numMelBins) }

        // Reusable buffers per frame.
        val windowed = FloatArray(frameLength)
        val real = FloatArray(fftSize)
        val imag = FloatArray(fftSize)

        for (f in 0 until n) {
            extractFrame(audio, f, windowed)

            if (removeDcOffset) {
                var sum = 0f
                for (i in windowed.indices) sum += windowed[i]
                val mean = sum / frameLength
                for (i in windowed.indices) windowed[i] -= mean
            }

            preemphasize(windowed, preemphCoeff)

            // Apply window.
            for (i in 0 until frameLength) {
                windowed[i] *= window[i]
            }

            // Zero-pad into FFT input.
            for (i in 0 until frameLength) {
                real[i] = windowed[i]
                imag[i] = 0f
            }
            for (i in frameLength until fftSize) {
                real[i] = 0f
                imag[i] = 0f
            }

            fftInPlace(real, imag)

            // Power spectrum across N/2+1 bins.
            val numBins = fftSize / 2 + 1
            val power = FloatArray(numBins)
            for (k in 0 until numBins) {
                power[k] = real[k] * real[k] + imag[k] * imag[k]
            }

            // Mel filter bank → log.
            for (m in 0 until numMelBins) {
                val filt = melFilters[m]
                var sum = 0f
                for (k in 0 until numBins) {
                    sum += filt[k] * power[k]
                }
                out[f][m] = ln(max(sum, ENERGY_FLOOR))
            }
        }
        return out
    }

    /**
     * First-sample-index of the f-th frame (kaldi snip_edges = false).
     * May be negative; reflection padding handles that during extraction.
     */
    private fun firstSampleOfFrame(f: Int): Int {
        val midpoint = frameShift * f + frameShift / 2
        return midpoint - frameLength / 2
    }

    private fun extractFrame(audio: FloatArray, frameIndex: Int, dst: FloatArray) {
        val start = firstSampleOfFrame(frameIndex)
        val n = audio.size
        for (i in 0 until frameLength) {
            val srcIdx = start + i
            dst[i] = when {
                srcIdx < 0 -> {
                    // Kaldi reflection: index `j` becomes `-j - 1` when negative.
                    val refl = -srcIdx - 1
                    if (refl < n) audio[refl] else 0f
                }
                srcIdx >= n -> {
                    val refl = 2 * n - srcIdx - 1
                    if (refl in 0 until n) audio[refl] else 0f
                }
                else -> audio[srcIdx]
            }
        }
    }

    private fun preemphasize(buf: FloatArray, coeff: Float) {
        if (coeff == 0f) return
        for (i in buf.size - 1 downTo 1) {
            buf[i] -= coeff * buf[i - 1]
        }
        buf[0] -= coeff * buf[0]
    }

    companion object {
        private const val ENERGY_FLOOR: Float = 1e-10f

        private fun nextPowerOfTwo(n: Int): Int {
            var p = 1
            while (p < n) p = p shl 1
            return p
        }

        private fun poveyWindow(length: Int): FloatArray {
            val w = FloatArray(length)
            val a = 2.0 * PI / (length - 1)
            for (i in 0 until length) {
                val v = 0.5 - 0.5 * cos(a * i)
                w[i] = v.pow(0.85).toFloat()
            }
            return w
        }

        // HTK mel scale (kaldi default).
        private fun hzToMel(hz: Float): Float = 1127f * ln(1f + hz / 700f)
        private fun melToHz(mel: Float): Float = 700f * (exp(mel / 1127f) - 1f)

        /**
         * Build [numMelBins] triangular filters over an `fftSize/2 + 1` bin spectrum.
         * Filter weights are returned in [filter][bin] form.
         */
        private fun buildMelFilters(
            fftSize: Int,
            sampleRate: Int,
            numMelBins: Int,
            lowFreq: Float,
            highFreq: Float,
        ): Array<FloatArray> {
            val numBins = fftSize / 2 + 1
            val melLow = hzToMel(lowFreq)
            val melHigh = hzToMel(highFreq)
            // numMelBins+2 mel-spaced points: lower edge, peaks, upper edge.
            val melPoints = FloatArray(numMelBins + 2) { i ->
                melLow + (melHigh - melLow) * i / (numMelBins + 1)
            }
            // Convert each mel point to fft-bin index (float; we interpolate weights).
            val binCenters = FloatArray(numMelBins + 2) { i ->
                melToHz(melPoints[i]) * fftSize / sampleRate
            }
            val filters = Array(numMelBins) { FloatArray(numBins) }
            for (m in 0 until numMelBins) {
                val left = binCenters[m]
                val center = binCenters[m + 1]
                val right = binCenters[m + 2]
                for (k in 0 until numBins) {
                    val kf = k.toFloat()
                    val w = when {
                        kf < left -> 0f
                        kf < center -> (kf - left) / (center - left)
                        kf < right -> (right - kf) / (right - center)
                        else -> 0f
                    }
                    filters[m][k] = w
                }
            }
            return filters
        }

        /**
         * In-place radix-2 iterative Cooley–Tukey FFT. Length must be a power of 2.
         */
        private fun fftInPlace(real: FloatArray, imag: FloatArray) {
            val n = real.size
            require(n > 0 && (n and (n - 1)) == 0) { "FFT size must be power of 2" }

            // Bit-reverse permutation.
            var j = 0
            for (i in 1 until n) {
                var bit = n shr 1
                while (j and bit != 0) {
                    j = j xor bit
                    bit = bit shr 1
                }
                j = j or bit
                if (i < j) {
                    val tr = real[i]; real[i] = real[j]; real[j] = tr
                    val ti = imag[i]; imag[i] = imag[j]; imag[j] = ti
                }
            }

            // Cooley–Tukey butterflies.
            var len = 2
            while (len <= n) {
                val angle = -2.0 * PI / len
                val wLenR = cos(angle).toFloat()
                val wLenI = sin(angle).toFloat()
                var i = 0
                while (i < n) {
                    var wR = 1f
                    var wI = 0f
                    val half = len / 2
                    for (k in 0 until half) {
                        val uR = real[i + k]
                        val uI = imag[i + k]
                        val tR = real[i + k + half] * wR - imag[i + k + half] * wI
                        val tI = real[i + k + half] * wI + imag[i + k + half] * wR
                        real[i + k] = uR + tR
                        imag[i + k] = uI + tI
                        real[i + k + half] = uR - tR
                        imag[i + k + half] = uI - tI
                        val newWR = wR * wLenR - wI * wLenI
                        wI = wR * wLenI + wI * wLenR
                        wR = newWR
                    }
                    i += len
                }
                len = len shl 1
            }
        }

        // Suppress unused-import lint hints for math fns we may want later.
        @Suppress("unused")
        private fun unused(): Double = log10(0.0) + sqrt(0.0)
    }
}
