package dev.soupslurpr.transcribro.recognitionservice.remote

import org.json.JSONException
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.net.URI
import java.net.URISyntaxException

/*
 * Pure-JVM building blocks for talking to an OpenAI-compatible
 * `/v1/audio/transcriptions` endpoint. Nothing in this file touches Android APIs, so
 * the encoding / parsing rules can be covered by plain JUnit tests on the host JVM
 * (org.json is supplied there by a test dependency; on device it comes from the
 * framework).
 */

/** Connection settings for a user-configured transcription server. */
data class RemoteServerConfig(
    /** Full endpoint URL, used exactly as entered. We never append a path to it. */
    val url: String,
    /** Value for the `model` form field. Omitted from the request when null/blank. */
    val model: String? = null,
    /** Sent as `Authorization: Bearer <token>` when non-blank. */
    val bearerToken: String? = null,
)

/** Result of [validateServerUrl]. */
sealed class ServerUrlValidation {
    /** [url] is the trimmed input, ready to be passed to the HTTP client. */
    data class Valid(val url: String) : ServerUrlValidation()

    /** [reason] is short and user-facing; it's shown under the settings field. */
    data class Invalid(val reason: String) : ServerUrlValidation()
}

/**
 * Accept only absolute `https://` URLs with a host. Cleartext is rejected here on
 * purpose rather than relying solely on the platform's cleartext block: audio of the
 * user's voice should never travel unencrypted, even on a "trusted" LAN or tailnet, and
 * a clear message in settings beats an opaque CLEARTEXT_NOT_PERMITTED at record time.
 */
fun validateServerUrl(input: String): ServerUrlValidation {
    val trimmed = input.trim()
    if (trimmed.isEmpty()) return ServerUrlValidation.Invalid("Enter the server's transcription URL")
    if (trimmed.any { it.isWhitespace() }) {
        return ServerUrlValidation.Invalid("URL must not contain spaces")
    }
    val uri = try {
        URI(trimmed)
    } catch (e: URISyntaxException) {
        return ServerUrlValidation.Invalid("Not a valid URL")
    }
    val scheme = uri.scheme?.lowercase()
    if (scheme == null) return ServerUrlValidation.Invalid("URL must start with https://")
    if (scheme != "https") return ServerUrlValidation.Invalid("Only https:// URLs are allowed")
    if (uri.host.isNullOrEmpty()) return ServerUrlValidation.Invalid("URL is missing a host name")
    return ServerUrlValidation.Valid(trimmed)
}

/**
 * Encode mono 16-bit PCM [samples] as a canonical 44-byte-header RIFF/WAVE file. This
 * is the format whisper itself consumes, so the server doesn't need to resample.
 */
fun encodeWav(samples: ShortArray, sampleRate: Int = 16000): ByteArray {
    val channels = 1
    val bitsPerSample = 16
    val byteRate = sampleRate * channels * bitsPerSample / 8
    val blockAlign = channels * bitsPerSample / 8
    val dataSize = samples.size * 2
    val out = ByteArray(44 + dataSize)
    var p = 0

    fun ascii(s: String) {
        for (c in s) out[p++] = c.code.toByte()
    }

    fun le32(v: Int) {
        out[p++] = (v and 0xFF).toByte()
        out[p++] = ((v ushr 8) and 0xFF).toByte()
        out[p++] = ((v ushr 16) and 0xFF).toByte()
        out[p++] = ((v ushr 24) and 0xFF).toByte()
    }

    fun le16(v: Int) {
        out[p++] = (v and 0xFF).toByte()
        out[p++] = ((v ushr 8) and 0xFF).toByte()
    }

    ascii("RIFF")
    le32(36 + dataSize)
    ascii("WAVE")
    ascii("fmt ")
    le32(16) // PCM fmt chunk size
    le16(1) // audio format: PCM
    le16(channels)
    le32(sampleRate)
    le32(byteRate)
    le16(blockAlign)
    le16(bitsPerSample)
    ascii("data")
    le32(dataSize)
    for (s in samples) le16(s.toInt())
    return out
}

/** A fully-serialized multipart/form-data request body plus its matching header value. */
class MultipartBody(val contentType: String, val bytes: ByteArray)

/**
 * Serialize a multipart/form-data body with plain text [fields] followed by a single
 * file part. Field order is preserved; some server implementations stream-parse and
 * read options before the (large) file, so text fields go first.
 */
fun buildMultipartBody(
    boundary: String,
    fields: List<Pair<String, String>>,
    fileFieldName: String,
    fileName: String,
    fileContentType: String,
    fileBytes: ByteArray,
): MultipartBody {
    val out = ByteArrayOutputStream(fileBytes.size + 512)
    fun write(s: String) = out.write(s.toByteArray(Charsets.UTF_8))

    for ((name, value) in fields) {
        write("--$boundary\r\n")
        write("Content-Disposition: form-data; name=\"$name\"\r\n\r\n")
        write(value)
        write("\r\n")
    }
    write("--$boundary\r\n")
    write("Content-Disposition: form-data; name=\"$fileFieldName\"; filename=\"$fileName\"\r\n")
    write("Content-Type: $fileContentType\r\n\r\n")
    out.write(fileBytes)
    write("\r\n--$boundary--\r\n")
    return MultipartBody("multipart/form-data; boundary=$boundary", out.toByteArray())
}

/**
 * The text form fields for a transcription request. `language` is pinned to English
 * because the on-device path is English-only today (jni.c hardcodes "en") and both
 * backends should behave the same; `model` is only sent when the user set one, so
 * servers that host a single model don't reject an unknown name.
 */
fun transcriptionFormFields(config: RemoteServerConfig): List<Pair<String, String>> = buildList {
    add("response_format" to "json")
    add("language" to "en")
    val model = config.model?.trim()
    if (!model.isNullOrEmpty()) add("model" to model)
}

/** Thrown for any failed request; [httpStatus] is null for transport-level failures. */
class RemoteTranscriptionException(
    message: String,
    val httpStatus: Int? = null,
    cause: Throwable? = null,
) : Exception(message, cause)

/**
 * Pull `text` out of a successful JSON reply. Anything else the server adds (segments,
 * usage, duration) is ignored. A reply without a `text` string is treated as an error
 * rather than an empty transcript so a misconfigured URL (e.g. pointing at some other
 * JSON API) doesn't silently swallow the user's speech.
 */
fun parseTranscriptionResponse(body: String): String {
    val json = try {
        JSONObject(body)
    } catch (e: JSONException) {
        throw RemoteTranscriptionException("Server reply was not JSON", cause = e)
    }
    if (!json.has("text") || json.isNull("text")) {
        throw RemoteTranscriptionException("Server reply had no \"text\" field")
    }
    return json.optString("text", "")
}

/**
 * Best-effort human-readable message for a non-2xx reply. Prefers the OpenAI shape
 * `{"error": {"message": ...}}`, tolerates `{"error": "..."}` and FastAPI's
 * `{"detail": "..."}`, and otherwise falls back to a truncated raw body so the user at
 * least sees what came back (an HTML proxy error page, for instance).
 */
fun parseErrorMessage(body: String?): String? {
    if (body.isNullOrBlank()) return null
    try {
        val json = JSONObject(body)
        val error = json.opt("error")
        if (error is JSONObject) {
            val msg = error.optString("message", "")
            if (msg.isNotBlank()) return msg
        } else if (error is String && error.isNotBlank()) {
            return error
        }
        val detail = json.opt("detail")
        if (detail is String && detail.isNotBlank()) return detail
    } catch (_: JSONException) {
        // Not JSON — fall through to the raw body.
    }
    val raw = body.trim()
    return if (raw.length > MAX_RAW_ERROR_CHARS) raw.take(MAX_RAW_ERROR_CHARS) + "…" else raw
}

private const val MAX_RAW_ERROR_CHARS = 200

/**
 * Bring server text into the same shape as the on-device whisper output (see
 * [dev.soupslurpr.transcribro.recognitionservice.TranscriptionEngine]): trim, strip the
 * trailing " ." whisper hallucinates on near-silence, and re-add a single leading space
 * so the keyboard's segment-joining logic keeps working.
 */
fun normalizeRemoteTranscript(text: String): String {
    val t = text.trim().removeSuffix(" .").trim()
    if (t.isEmpty() || t == ".") return ""
    return " $t"
}
