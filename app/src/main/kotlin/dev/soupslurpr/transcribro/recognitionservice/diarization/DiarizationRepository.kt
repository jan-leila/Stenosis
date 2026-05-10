package dev.soupslurpr.transcribro.recognitionservice.diarization

/**
 * Owns the pyannote segmenter + wespeaker embedder ONNX sessions and lends them to
 * callers (typically [DiarizationManager]) on demand. Both models are loaded lazily on
 * first access — a session that never enables diarization never pays the ~30 MB +
 * ONNX-runtime-init cost.
 *
 * Both models are stateless from the registry's perspective: the LSTM hidden state is
 * carried inside the segmenter window-by-window, but inference itself is a pure
 * function of the input. So a single segmenter / embedder instance can be reused across
 * recognition sessions; we only release on service teardown.
 */
class DiarizationRepository(
    private val segmenterDataSource: PyannoteSegmenterLocalDataSource,
    private val embedderDataSource: PyannoteEmbedderLocalDataSource,
) {
    private var segmenter: PyannoteSegmenterOnnxModel? = null
    private var embedder: PyannoteEmbedderOnnxModel? = null

    suspend fun getSegmenter(): PyannoteSegmenterOnnxModel =
        segmenter ?: segmenterDataSource.getSegmenter().also { segmenter = it }

    suspend fun getEmbedder(): PyannoteEmbedderOnnxModel =
        embedder ?: embedderDataSource.getEmbedder().also { embedder = it }

    /** Eagerly load both models (e.g. at start-of-session if diarization is enabled). */
    suspend fun preload() {
        getSegmenter()
        getEmbedder()
    }

    fun release() {
        segmenter?.close()
        segmenter = null
        embedder?.close()
        embedder = null
    }
}
