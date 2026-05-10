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

    line("")
    runSpeakerRegistryTest(::line)
    line("")
    runSpeakerLabelStoreTest(::line)

    log.toString().trimEnd()
}

/**
 * Smoke-test [SpeakerRegistry] with synthetic embeddings. Random 256-dim unit vectors
 * are approximately orthogonal so they cluster as separate speakers; small jitter
 * around a centroid stays within the same-speaker threshold. Deterministic via seed.
 */
private fun runSpeakerRegistryTest(line: (String) -> Unit) {
    line("[registry] testing SpeakerRegistry with synthetic embeddings...")
    val registry = SpeakerRegistry()
    val rng = java.util.Random(42L)

    val speakerA = randomUnitVector(PyannoteEmbedderOnnxModel.EMBEDDING_DIM, rng)
    val speakerB = randomUnitVector(PyannoteEmbedderOnnxModel.EMBEDDING_DIM, rng)
    val baseDist = SpeakerRegistry.cosineDistance(speakerA, speakerB)
    line("[registry]   A vs B cosine distance: ${"%.3f".format(baseDist)} (random 256-dim vectors are ≈orthogonal ≈1.0)")

    // sigma=0.05 puts pairwise same-speaker cosine distance at ~0.4 — close to the
    // 0.5 threshold but reliably under it, which mirrors what real wespeaker
    // same-speaker pairs look like (~0.1–0.3 in practice). At sigma=0.1 the variance
    // pushes pairwise distances above 0.7 and they stop clustering, which is a test-
    // setup artifact, not a registry bug.
    val aSamples = (0 until 3).map { jitter(speakerA, 0.05f, rng) }
    val bSamples = (0 until 3).map { jitter(speakerB, 0.05f, rng) }

    // Sanity-check the synthetic geometry — if these are above the threshold the
    // clustering test will fail and we want to know why before chasing a bug in
    // SpeakerRegistry itself.
    val sameSpeakerDist = SpeakerRegistry.cosineDistance(aSamples[0], aSamples[1])
    val crossSpeakerDist = SpeakerRegistry.cosineDistance(aSamples[0], bSamples[0])
    line("[registry]   same-speaker pair distance: ${"%.3f".format(sameSpeakerDist)} (must be < ${"%.2f".format(SpeakerRegistry.DEFAULT_THRESHOLD)} for the test to pass)")
    line("[registry]   cross-speaker pair distance: ${"%.3f".format(crossSpeakerDist)} (must be > ${"%.2f".format(SpeakerRegistry.DEFAULT_THRESHOLD)} for the test to pass)")

    val ids = listOf(
        registry.assign(aSamples[0]),
        registry.assign(bSamples[0]),
        registry.assign(aSamples[1]),
        registry.assign(bSamples[1]),
        registry.assign(aSamples[2]),
        registry.assign(bSamples[2]),
    )
    val expected = listOf(0, 1, 0, 1, 0, 1)
    val clusterOk = ids == expected
    line("[registry]   assigned ids: $ids (expected $expected) ${if (clusterOk) "✓" else "✗"}")
    line("[registry]   total speakers: ${registry.size} (expected 2)")

    registry.reset()
    line("[registry]   after reset: size=${registry.size} (expected 0)")
}

private fun runSpeakerLabelStoreTest(line: (String) -> Unit) {
    line("[labels] testing SpeakerLabelStore...")
    val store = SpeakerLabelStore()
    line("[labels]   default(0)='${store.displayName(0)}' (expected 'Speaker 1')")
    line("[labels]   default(2)='${store.displayName(2)}' (expected 'Speaker 3')")
    store.rename(0, "Alice")
    line("[labels]   rename(0, 'Alice') ⇒ default(0)='${store.displayName(0)}'")
    store.bulkRename(mapOf(1 to "Bob", 2 to "Carol"))
    line("[labels]   bulkRename: 0='${store.displayName(0)}' 1='${store.displayName(1)}' 2='${store.displayName(2)}' 3='${store.displayName(3)}'")
    store.reset()
    line("[labels]   after reset: default(0)='${store.displayName(0)}' (expected 'Speaker 1')")
}

private fun randomUnitVector(dim: Int, rng: java.util.Random): FloatArray {
    val v = FloatArray(dim) { rng.nextGaussian().toFloat() }
    var sumSq = 0.0
    for (x in v) sumSq += x * x
    val norm = sqrt(sumSq).toFloat()
    if (norm > 0f) for (i in v.indices) v[i] /= norm
    return v
}

private fun jitter(base: FloatArray, sigma: Float, rng: java.util.Random): FloatArray {
    val v = FloatArray(base.size) { base[it] + (rng.nextGaussian() * sigma).toFloat() }
    var sumSq = 0.0
    for (x in v) sumSq += x * x
    val norm = sqrt(sumSq).toFloat()
    if (norm > 0f) for (i in v.indices) v[i] /= norm
    return v
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
