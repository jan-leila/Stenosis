package dev.soupslurpr.transcribro.recognitionservice.diarization

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.withContext

/**
 * Decoupled factory for [PyannoteSegmenterOnnxModel] so the service layer can be unit-tested
 * with a fake. Mirrors the SileroVadApi / WhisperApi pattern.
 */
interface PyannoteSegmenterApi {
    fun getSegmenter(): PyannoteSegmenterOnnxModel
}

class PyannoteSegmenterLocalDataSource(
    private val api: PyannoteSegmenterApi,
    private val ioDispatcher: CoroutineDispatcher,
) {
    suspend fun getSegmenter(): PyannoteSegmenterOnnxModel =
        withContext(ioDispatcher) { api.getSegmenter() }
}
