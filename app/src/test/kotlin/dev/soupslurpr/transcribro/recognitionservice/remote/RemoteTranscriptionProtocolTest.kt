package dev.soupslurpr.transcribro.recognitionservice.remote

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class RemoteTranscriptionProtocolTest {

    // --- WAV -------------------------------------------------------------------------

    @Test
    fun wavHeaderMatchesCanonicalPcmLayout() {
        val samples = shortArrayOf(0, 1, -1, Short.MAX_VALUE, Short.MIN_VALUE)
        val wav = encodeWav(samples, sampleRate = 16000)
        val buf = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN)

        assertEquals(44 + samples.size * 2, wav.size)
        assertEquals("RIFF", String(wav, 0, 4, Charsets.US_ASCII))
        assertEquals(36 + samples.size * 2, buf.getInt(4))
        assertEquals("WAVE", String(wav, 8, 4, Charsets.US_ASCII))
        assertEquals("fmt ", String(wav, 12, 4, Charsets.US_ASCII))
        assertEquals(16, buf.getInt(16)) // fmt chunk size
        assertEquals(1, buf.getShort(20).toInt()) // PCM
        assertEquals(1, buf.getShort(22).toInt()) // mono
        assertEquals(16000, buf.getInt(24)) // sample rate
        assertEquals(32000, buf.getInt(28)) // byte rate
        assertEquals(2, buf.getShort(32).toInt()) // block align
        assertEquals(16, buf.getShort(34).toInt()) // bits per sample
        assertEquals("data", String(wav, 36, 4, Charsets.US_ASCII))
        assertEquals(samples.size * 2, buf.getInt(40))
    }

    @Test
    fun wavSamplesAreLittleEndianSigned() {
        val samples = shortArrayOf(0x1234, -2, Short.MIN_VALUE)
        val wav = encodeWav(samples)
        assertArrayEquals(
            byteArrayOf(0x34, 0x12, 0xFE.toByte(), 0xFF.toByte(), 0x00, 0x80.toByte()),
            wav.copyOfRange(44, wav.size),
        )
    }

    @Test
    fun wavOfNoSamplesIsHeaderOnly() {
        val wav = encodeWav(ShortArray(0))
        val buf = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN)
        assertEquals(44, wav.size)
        assertEquals(36, buf.getInt(4))
        assertEquals(0, buf.getInt(40))
    }

    // --- Multipart -------------------------------------------------------------------

    @Test
    fun multipartBodyIsSerializedExactly() {
        val body = buildMultipartBody(
            boundary = "BOUNDARY",
            fields = listOf("response_format" to "json", "language" to "en"),
            fileFieldName = "file",
            fileName = "audio.wav",
            fileContentType = "audio/wav",
            fileBytes = byteArrayOf(1, 2, 3),
        )
        assertEquals("multipart/form-data; boundary=BOUNDARY", body.contentType)

        val expected = (
            "--BOUNDARY\r\n" +
                "Content-Disposition: form-data; name=\"response_format\"\r\n\r\n" +
                "json\r\n" +
                "--BOUNDARY\r\n" +
                "Content-Disposition: form-data; name=\"language\"\r\n\r\n" +
                "en\r\n" +
                "--BOUNDARY\r\n" +
                "Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n" +
                "Content-Type: audio/wav\r\n\r\n"
            ).toByteArray(Charsets.UTF_8) +
            byteArrayOf(1, 2, 3) +
            "\r\n--BOUNDARY--\r\n".toByteArray(Charsets.UTF_8)
        assertArrayEquals(expected, body.bytes)
    }

    @Test
    fun multipartBodyPreservesBinaryFileBytes() {
        // Every byte value, so a charset round-trip anywhere in the builder would show up.
        val file = ByteArray(256) { it.toByte() }
        val body = buildMultipartBody("b", emptyList(), "file", "a.wav", "audio/wav", file)
        val header = "--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n" +
            "Content-Type: audio/wav\r\n\r\n"
        val start = header.toByteArray(Charsets.UTF_8).size
        assertArrayEquals(file, body.bytes.copyOfRange(start, start + file.size))
    }

    @Test
    fun formFieldsOmitModelWhenBlank() {
        assertEquals(
            listOf("response_format" to "json", "language" to "en"),
            transcriptionFormFields(RemoteServerConfig(url = "https://x", model = null)),
        )
        assertEquals(
            listOf("response_format" to "json", "language" to "en"),
            transcriptionFormFields(RemoteServerConfig(url = "https://x", model = "   ")),
        )
    }

    @Test
    fun formFieldsIncludeTrimmedModelWhenSet() {
        assertEquals(
            listOf("response_format" to "json", "language" to "en", "model" to "whisper-1"),
            transcriptionFormFields(RemoteServerConfig(url = "https://x", model = " whisper-1 ")),
        )
    }

    // --- Response parsing ------------------------------------------------------------

    @Test
    fun parsesTextAndIgnoresOtherFields() {
        assertEquals(
            " Hello there.",
            parseTranscriptionResponse("""{"text":" Hello there.","duration":1.5,"segments":[]}"""),
        )
    }

    @Test
    fun emptyTextIsAValidTranscript() {
        assertEquals("", parseTranscriptionResponse("""{"text":""}"""))
    }

    @Test
    fun missingTextIsAnError() {
        expectRemoteError { parseTranscriptionResponse("""{"transcript":"hi"}""") }
        expectRemoteError { parseTranscriptionResponse("""{"text":null}""") }
    }

    @Test
    fun nonJsonReplyIsAnError() {
        expectRemoteError { parseTranscriptionResponse("<html>502 Bad Gateway</html>") }
    }

    @Test
    fun errorMessageFromOpenAiShape() {
        assertEquals(
            "File too large",
            parseErrorMessage("""{"error":{"message":"File too large","type":"invalid_request_error"}}"""),
        )
    }

    @Test
    fun errorMessageFromAlternateShapes() {
        assertEquals("nope", parseErrorMessage("""{"error":"nope"}"""))
        assertEquals("Not Found", parseErrorMessage("""{"detail":"Not Found"}"""))
    }

    @Test
    fun errorMessageFallsBackToRawBody() {
        assertEquals("Bad Gateway", parseErrorMessage("  Bad Gateway \n"))
        val long = "x".repeat(500)
        val msg = parseErrorMessage(long)!!
        assertEquals(201, msg.length)
        assertTrue(msg.endsWith("…"))
    }

    @Test
    fun errorMessageOfBlankBodyIsNull() {
        assertNull(parseErrorMessage(null))
        assertNull(parseErrorMessage("  "))
    }

    @Test
    fun normalizeMatchesWhisperOutputShape() {
        assertEquals(" Hello world.", normalizeRemoteTranscript("Hello world."))
        assertEquals(" Hello world.", normalizeRemoteTranscript("  Hello world.\n"))
        assertEquals(" Hello", normalizeRemoteTranscript("Hello ."))
        assertEquals("", normalizeRemoteTranscript(""))
        assertEquals("", normalizeRemoteTranscript("   "))
        assertEquals("", normalizeRemoteTranscript(" ."))
    }

    // --- URL validation --------------------------------------------------------------

    @Test
    fun acceptsHttpsUrlsVerbatimAfterTrim() {
        assertEquals(
            ServerUrlValidation.Valid("https://transcription.example.com/v1/audio/transcriptions"),
            validateServerUrl("  https://transcription.example.com/v1/audio/transcriptions "),
        )
        assertEquals(
            ServerUrlValidation.Valid("HTTPS://host:8443/x?y=1"),
            validateServerUrl("HTTPS://host:8443/x?y=1"),
        )
    }

    @Test
    fun rejectsNonHttpsAndMalformedUrls() {
        for (bad in listOf(
            "",
            "   ",
            "http://host/v1/audio/transcriptions",
            "ftp://host/file",
            "host/v1/audio/transcriptions",
            "https://",
            "https:///path",
            "https://ho st/path",
            "https://host/pa th",
            "https://host/%zz",
        )) {
            val result = validateServerUrl(bad)
            assertTrue("expected \"$bad\" to be rejected, got $result", result is ServerUrlValidation.Invalid)
        }
    }

    private fun expectRemoteError(block: () -> Unit) {
        try {
            block()
            fail("expected RemoteTranscriptionException")
        } catch (_: RemoteTranscriptionException) {
        }
    }
}
