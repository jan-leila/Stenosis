package dev.soupslurpr.transcribro.recognitionservice.diarization

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.FileNotFoundException
import kotlin.math.PI
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * One-shot smoke test for the pyannote ONNX wrappers. Loads both model files from
 * assets, runs them on synthetic audio, returns a human-readable report.
 *
 * Catches the typical phase-1 failure modes:
 *   - missing asset (asset paths wrong, or models not yet downloaded)
 *   - ONNX session creation errors (corrupt file, wrong export, version mismatch)
 *   - I/O name mismatches (sherpa-onnx export vs other community exports)
 *   - shape mismatches in the wrapper
 *
 * Latency printed below is just one cold inference each — useful for sanity but not a
 * benchmark.
 */
suspend fun verifyDiarizationModels(context: Context): String = withContext(Dispatchers.IO) {
    val log = StringBuilder()
    fun line(msg: String) {
        log.appendLine(msg)
        Log.i(TAG, msg)
    }

    // --- segmenter -----------------------------------------------------------------
    line("[segmenter] loading ${DiarizationAssets.SEGMENTATION_ASSET}")
    val segmenter: PyannoteSegmenterOnnxModel? = try {
        val bytes = context.assets.open(DiarizationAssets.SEGMENTATION_ASSET)
            .use { it.readBytes() }
        line("[segmenter] ${bytes.size} bytes")
        PyannoteSegmenterOnnxModel(bytes)
    } catch (e: FileNotFoundException) {
        line("[segmenter] MISSING — model wasn't packaged into the APK.")
        line("[segmenter] The downloadDiarizationModels Gradle task should fetch it on build;")
        line("[segmenter] try `./gradlew downloadDiarizationModels` to surface any network errors.")
        null
    } catch (e: Throwable) {
        line("[segmenter] LOAD FAILED ${e::class.simpleName}: ${e.message}")
        Log.e(TAG, "segmenter load failed", e)
        null
    }

    if (segmenter != null) {
        try {
            val silent = FloatArray(PyannoteSegmenterOnnxModel.WINDOW_SAMPLES)
            val t0 = System.currentTimeMillis()
            val resSilent = segmenter.segment(silent)
            val dtSilent = System.currentTimeMillis() - t0
            val activeFramesSilent = (0 until resSilent.numFrames).count { resSilent.isAnySpeechAt(it) }
            line("[segmenter] silence pass ${dtSilent}ms")
            line("[segmenter]   frames=${resSilent.numFrames} (expected ${PyannoteSegmenterOnnxModel.NUM_FRAMES})")
            line("[segmenter]   localSpeakers=${resSilent.numLocalSpeakers} (expected ${PyannoteSegmenterOnnxModel.NUM_LOCAL_SPEAKERS})")
            line("[segmenter]   active frames on silence: $activeFramesSilent (expected ≈0)")

            // A 1kHz tone isn't speech, but lets us check that something other than the
            // silent class can win on non-trivial input.
            val tone = FloatArray(PyannoteSegmenterOnnxModel.WINDOW_SAMPLES) { i ->
                (0.3 * sin(2 * PI * 1000.0 * i / PyannoteSegmenterOnnxModel.SAMPLE_RATE)).toFloat()
            }
            val t1 = System.currentTimeMillis()
            val resTone = segmenter.segment(tone)
            val dtTone = System.currentTimeMillis() - t1
            val activeFramesTone = (0 until resTone.numFrames).count { resTone.isAnySpeechAt(it) }
            line("[segmenter] tone pass ${dtTone}ms")
            line("[segmenter]   active frames on 1kHz tone: $activeFramesTone")
        } catch (e: Throwable) {
            line("[segmenter] INFERENCE FAILED ${e::class.simpleName}: ${e.message}")
            Log.e(TAG, "segmenter inference failed", e)
        } finally {
            try { segmenter.close() } catch (_: Throwable) {}
        }
    }

    line("")

    // --- embedder ------------------------------------------------------------------
    line("[embedder] loading ${DiarizationAssets.EMBEDDER_ASSET}")
    val embedder: PyannoteEmbedderOnnxModel? = try {
        val bytes = context.assets.open(DiarizationAssets.EMBEDDER_ASSET)
            .use { it.readBytes() }
        line("[embedder] ${bytes.size} bytes")
        PyannoteEmbedderOnnxModel(bytes)
    } catch (e: FileNotFoundException) {
        line("[embedder] MISSING — model wasn't packaged into the APK.")
        line("[embedder] The downloadDiarizationModels Gradle task should fetch it on build;")
        line("[embedder] try `./gradlew downloadDiarizationModels` to surface any network errors.")
        null
    } catch (e: Throwable) {
        line("[embedder] LOAD FAILED ${e::class.simpleName}: ${e.message}")
        Log.e(TAG, "embedder load failed", e)
        null
    }

    if (embedder != null) {
        try {
            // 1.5s of a 440Hz tone in int16 amplitude range (NOT [-1,1]).
            val sr = PyannoteEmbedderOnnxModel.SAMPLE_RATE
            val n = sr * 3 / 2
            val tone440 = FloatArray(n) { i ->
                (8000.0 * sin(2 * PI * 440.0 * i / sr)).toFloat()
            }
            val tone880 = FloatArray(n) { i ->
                (8000.0 * sin(2 * PI * 880.0 * i / sr)).toFloat()
            }

            val t0 = System.currentTimeMillis()
            val emb1 = embedder.embed(tone440)
            val dt0 = System.currentTimeMillis() - t0
            val t1 = System.currentTimeMillis()
            val emb2 = embedder.embed(tone880)
            val dt1 = System.currentTimeMillis() - t1

            line("[embedder] inference ${dt0}ms / ${dt1}ms")
            line("[embedder]   dim=${emb1.size} (expected ${PyannoteEmbedderOnnxModel.EMBEDDING_DIM})")
            line("[embedder]   ‖emb1‖₂=${"%.4f".format(l2Norm(emb1))} (expected ≈1)")
            val cos = cosineSimilarity(emb1, emb2)
            line("[embedder]   cos(440Hz, 880Hz)=${"%.3f".format(cos)} (expected <1; tones aren't speech so don't read too much into it)")

            // Same audio twice should yield ≈identical embedding — this is the strongest
            // sanity check that the pipeline is deterministic.
            val emb1b = embedder.embed(tone440)
            val cosSelf = cosineSimilarity(emb1, emb1b)
            line("[embedder]   cos(emb(audio), emb(same audio))=${"%.4f".format(cosSelf)} (expected ≈1.0)")
        } catch (e: Throwable) {
            line("[embedder] INFERENCE FAILED ${e::class.simpleName}: ${e.message}")
            Log.e(TAG, "embedder inference failed", e)
        } finally {
            try { embedder.close() } catch (_: Throwable) {}
        }
    }

    log.toString().trimEnd()
}

private fun l2Norm(v: FloatArray): Double {
    var s = 0.0
    for (x in v) s += x * x
    return sqrt(s)
}

private fun cosineSimilarity(a: FloatArray, b: FloatArray): Double {
    require(a.size == b.size)
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    for (i in a.indices) {
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    }
    return dot / (sqrt(na) * sqrt(nb) + 1e-12)
}

private const val TAG = "DiarizationVerify"
