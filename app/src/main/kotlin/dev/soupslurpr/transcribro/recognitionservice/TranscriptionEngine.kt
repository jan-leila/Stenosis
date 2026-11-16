package dev.soupslurpr.transcribro.recognitionservice

import kotlinx.coroutines.CancellationException

/**
 * The segment-final ("high quality") transcription pass, abstracted so the recognition
 * service can route it either to the on-device whisper.cpp context or to a self-hosted
 * server without the audio loop caring which one it is talking to.
 *
 * Only the final pass is abstracted. The streaming preview pass fires roughly once a
 * second and depends on whisper.cpp specifics (adaptive audio_ctx), so it stays bound
 * to [dev.soupslurpr.transcribro.recognitionservice.whisper.WhisperRepository] directly;
 * sending it over HTTP would mean a request per second per open segment, which is both
 * wasteful and too laggy to be useful as a "live" preview.
 *
 * Implementations return text in the same shape whisper.cpp does: a leading space
 * before the first word (the keyboard IME uses that leading space as the separator
 * between consecutive segments and strips it itself at the start of a line) and no
 * trailing " ." hallucination. Blank audio yields an empty string.
 */
interface TranscriptionEngine {
    suspend fun transcribeAudio(data: ShortArray): String
}

/**
 * Wraps a [primary] engine (the server) with an optional on-device [fallback]. When the
 * primary throws — network down, server returned non-2xx, read timeout — the same audio
 * is retried on [fallback] and [onFallback] is told why, so the service can let the user
 * know their audio was handled locally this time. With no fallback the primary's
 * exception propagates unchanged so the caller can surface it as a session error.
 *
 * Cancellation is never treated as a failure: if the session is cancelled while a
 * request is in flight, re-running the segment locally would just waste battery on a
 * result nobody will read.
 */
class FallbackTranscriptionEngine(
    private val primary: TranscriptionEngine,
    private val fallback: TranscriptionEngine?,
    private val onFallback: (Throwable) -> Unit = {},
) : TranscriptionEngine {
    override suspend fun transcribeAudio(data: ShortArray): String {
        return try {
            primary.transcribeAudio(data)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            if (fallback == null) throw e
            onFallback(e)
            fallback.transcribeAudio(data)
        }
    }
}
