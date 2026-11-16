package dev.soupslurpr.transcribro.recognitionservice.whisper

import com.whispercpp.whisper.WhisperContext
import dev.soupslurpr.transcribro.recognitionservice.TranscriptionEngine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class WhisperRepository(
    private val whisperLocalDataSource: WhisperLocalDataSource
) : TranscriptionEngine {
    // Two independent whisper contexts so the high-quality segment-final pass and the
    // cheap streaming preview pass can run concurrently. Each WhisperContext has its
    // own single-threaded coroutine dispatcher, so when a segment-final pass is in
    // flight on `finalContext`, the streaming preview pass on `streamingContext` keeps
    // running on its own thread instead of queueing behind it.
    //
    // Cost: tiny.en-q8 is loaded twice (~30MB of extra RAM). Worth it for live UX.
    //
    // Diarization-readiness note: keeping these separate also leaves room to later
    // route per-speaker streams to dedicated contexts (e.g. one per active speaker).
    private var streamingContext: WhisperContext? = null
    private var finalContext: WhisperContext? = null

    // Guards lazy context creation. With server transcription the final context is
    // only loaded on demand as a fallback, so several segment jobs that fail at once
    // could otherwise race here and each load (and leak) their own copy of the model.
    private val loadMutex = Mutex()

    private suspend fun ensureStreamingContext(): WhisperContext = loadMutex.withLock {
        streamingContext ?: whisperLocalDataSource.getStreamingContext().also { streamingContext = it }
    }

    private suspend fun ensureFinalContext(): WhisperContext = loadMutex.withLock {
        finalContext ?: whisperLocalDataSource.getFinalContext().also { finalContext = it }
    }

    /** Eagerly load both models so the first transcription doesn't pay model-load latency. */
    suspend fun preload() {
        preloadStreaming()
        preloadFinal()
    }

    /** Load only the streaming-preview context (no-op if already loaded). */
    suspend fun preloadStreaming() {
        ensureStreamingContext()
    }

    /** Load only the segment-final context (no-op if already loaded). */
    suspend fun preloadFinal() {
        ensureFinalContext()
    }

    override suspend fun transcribeAudio(data: ShortArray): String {
        val ctx = ensureFinalContext()
        // assume we only have one channel
        var buffer = FloatArray(data.size) { index ->
            (data[index] / 32767.0f).coerceIn(-1f..1f)
        }

        if (data.size < 32000) {
            val newBuffer = FloatArray(32000)

            for ((i, value) in buffer.withIndex()) {
                newBuffer[i] = value
            }

            newBuffer.fill(0f, data.size, newBuffer.size)

            buffer = newBuffer
        }

        val transcript = ctx.transcribeData(buffer, ((data.size / 16000f) * 1000f).toLong())
        return transcript.removeSuffix(" .") // remove hallucination
    }

    suspend fun transcribePartialAudio(data: ShortArray, audioCtx: Int? = null): String {
        val ctx = ensureStreamingContext()
        var buffer = FloatArray(data.size) { index ->
            (data[index] / 32767.0f).coerceIn(-1f..1f)
        }

        if (data.size < 32000) {
            val newBuffer = FloatArray(32000)
            for ((i, value) in buffer.withIndex()) {
                newBuffer[i] = value
            }
            newBuffer.fill(0f, data.size, newBuffer.size)
            buffer = newBuffer
        }

        val transcript = ctx.transcribeDataStreaming(
            buffer,
            ((data.size / 16000f) * 1000f).toLong(),
            audioCtx,
        )
        return transcript.removeSuffix(" .")
    }

    suspend fun release() = loadMutex.withLock {
        streamingContext?.release()
        streamingContext = null
        finalContext?.release()
        finalContext = null
    }
}