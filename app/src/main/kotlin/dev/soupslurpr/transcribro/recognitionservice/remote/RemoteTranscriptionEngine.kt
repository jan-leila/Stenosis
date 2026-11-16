package dev.soupslurpr.transcribro.recognitionservice.remote

import dev.soupslurpr.transcribro.recognitionservice.TranscriptionEngine
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.IOException
import java.net.HttpURLConnection
import java.net.SocketTimeoutException
import java.net.URL
import java.util.UUID

/**
 * Sends the segment-final pass to a self-hosted OpenAI-compatible transcription server.
 *
 * Uses [HttpURLConnection] for the same reason [dev.soupslurpr.transcribro.models.ModelInstaller]
 * does: one POST per finished segment doesn't justify pulling in an HTTP client
 * library, and the platform stack already does TLS with the system trust store (which
 * is what a publicly-issued ACME certificate needs).
 *
 * All network I/O runs on [ioDispatcher]; callers may invoke [transcribeAudio] from any
 * coroutine, including one on the main thread.
 */
class RemoteTranscriptionEngine(
    private val config: RemoteServerConfig,
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO,
    private val connectTimeoutMs: Int = DEFAULT_CONNECT_TIMEOUT_MS,
    private val readTimeoutMs: Int = DEFAULT_READ_TIMEOUT_MS,
) : TranscriptionEngine {

    override suspend fun transcribeAudio(data: ShortArray): String = withContext(ioDispatcher) {
        normalizeRemoteTranscript(postBlocking(data))
    }

    /**
     * Synchronous request; returns the raw `text` field. Throws
     * [RemoteTranscriptionException] for every failure mode so callers only need one
     * catch, with the original exception kept as the cause (a [SocketTimeoutException]
     * cause is how the service tells "timed out" apart from "unreachable").
     */
    internal fun postBlocking(data: ShortArray): String {
        val body = buildMultipartBody(
            boundary = "stenosis-" + UUID.randomUUID().toString(),
            fields = transcriptionFormFields(config),
            fileFieldName = "file",
            fileName = "audio.wav",
            fileContentType = "audio/wav",
            fileBytes = encodeWav(data),
        )

        val connection = try {
            (URL(config.url).openConnection() as HttpURLConnection)
        } catch (e: Exception) {
            throw RemoteTranscriptionException("Invalid server URL: ${e.describe()}", cause = e)
        }
        try {
            connection.requestMethod = "POST"
            connection.connectTimeout = connectTimeoutMs
            connection.readTimeout = readTimeoutMs
            connection.doOutput = true
            connection.useCaches = false
            // Redirects on a POST would silently turn into a GET on most stacks (and
            // drop the audio); better to report the misconfiguration than to "succeed"
            // against the wrong endpoint.
            connection.instanceFollowRedirects = false
            connection.setRequestProperty("Content-Type", body.contentType)
            connection.setRequestProperty("Accept", "application/json")
            connection.setRequestProperty("User-Agent", "Stenosis/1 (Android)")
            val token = config.bearerToken?.trim()
            if (!token.isNullOrEmpty()) {
                connection.setRequestProperty("Authorization", "Bearer $token")
            }
            connection.setFixedLengthStreamingMode(body.bytes.size)

            val status = try {
                connection.outputStream.use { it.write(body.bytes) }
                connection.responseCode
            } catch (e: SocketTimeoutException) {
                throw RemoteTranscriptionException("Server timed out", cause = e)
            } catch (e: IOException) {
                throw RemoteTranscriptionException("Could not reach server: ${e.describe()}", cause = e)
            }

            if (status !in 200..299) {
                val errorBody = try {
                    connection.errorStream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }
                } catch (_: IOException) {
                    null
                }
                val detail = parseErrorMessage(errorBody)
                throw RemoteTranscriptionException(
                    if (detail != null) "HTTP $status: $detail" else "HTTP $status",
                    httpStatus = status,
                )
            }

            val responseBody = try {
                connection.inputStream.bufferedReader(Charsets.UTF_8).use { it.readText() }
            } catch (e: SocketTimeoutException) {
                throw RemoteTranscriptionException("Server timed out", cause = e)
            } catch (e: IOException) {
                throw RemoteTranscriptionException("Failed reading server reply: ${e.describe()}", cause = e)
            }
            return parseTranscriptionResponse(responseBody)
        } finally {
            connection.disconnect()
        }
    }

    companion object {
        const val DEFAULT_CONNECT_TIMEOUT_MS = 10_000

        // A conversation segment only closes after the speaker pauses, so one request
        // can carry several minutes of audio; a CPU-only server may need a good while
        // to chew through that. Five minutes is generous without letting a hung server
        // wedge the session forever.
        const val DEFAULT_READ_TIMEOUT_MS = 5 * 60_000

        /** One second of silence: enough for a server to run a full request cycle. */
        val TEST_CONNECTION_AUDIO: ShortArray get() = ShortArray(16000)
    }
}

/** Compact "ClassName: message" for user-facing error strings. */
private fun Throwable.describe(): String {
    val name = this::class.simpleName ?: this::class.java.name
    val msg = message
    return if (msg.isNullOrBlank()) name else "$name: $msg"
}
