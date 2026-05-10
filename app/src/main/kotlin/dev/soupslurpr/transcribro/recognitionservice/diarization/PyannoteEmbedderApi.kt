package dev.soupslurpr.transcribro.recognitionservice.diarization

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.withContext

interface PyannoteEmbedderApi {
    fun getEmbedder(): PyannoteEmbedderOnnxModel
}

class PyannoteEmbedderLocalDataSource(
    private val api: PyannoteEmbedderApi,
    private val ioDispatcher: CoroutineDispatcher,
) {
    suspend fun getEmbedder(): PyannoteEmbedderOnnxModel =
        withContext(ioDispatcher) { api.getEmbedder() }
}
