package dev.soupslurpr.transcribro.recognitionservice.whisper

import com.whispercpp.whisper.WhisperContext
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.withContext

class WhisperLocalDataSource(
    private val whisperApi: WhisperApi,
    private val ioDispatcher: CoroutineDispatcher,
) {
    suspend fun getStreamingContext(): WhisperContext =
        withContext(ioDispatcher) { whisperApi.getStreamingContext() }

    suspend fun getFinalContext(): WhisperContext =
        withContext(ioDispatcher) { whisperApi.getFinalContext() }
}

/**
 * Lets a caller load two distinct whisper contexts — one for the segment-final pass and
 * one for the cheaper streaming-preview pass. If the caller wants the same model for
 * both, [getStreamingContext] and [getFinalContext] just return contexts built from the
 * same file (which is the current default).
 */
interface WhisperApi {
    fun getStreamingContext(): WhisperContext
    fun getFinalContext(): WhisperContext
}