package dev.soupslurpr.transcribro.recognitionservice.remote

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.net.SocketException
import java.net.SocketTimeoutException
import kotlin.concurrent.thread

/**
 * Drives [RemoteTranscriptionEngine] against a minimal in-process HTTP/1.1 server. Plain
 * http is fine here: the https-only rule lives in [validateServerUrl], which the service
 * and settings apply before an engine is ever built.
 *
 * The server is hand-rolled on [ServerSocket] because unit tests compile against
 * android.jar, which doesn't expose the JDK's com.sun.net.httpserver. It handles one
 * request per connection (`Connection: close`), which is all the engine ever sends.
 */
class RemoteTranscriptionEngineTest {
    private lateinit var server: ServerSocket

    // Captured from the last request the server saw.
    @Volatile private var method: String? = null
    @Volatile private var path: String? = null
    @Volatile private var contentType: String? = null
    @Volatile private var authorization: String? = null
    @Volatile private var body: ByteArray = ByteArray(0)

    // What the server replies with.
    @Volatile private var replyStatus = 200
    @Volatile private var replyBody = """{"text":"Hello world."}"""
    @Volatile private var replyDelayMs = 0L

    @Before
    fun setUp() {
        server = ServerSocket(0, 50, InetAddress.getByName("127.0.0.1"))
        thread(isDaemon = true) {
            while (true) {
                val socket = try {
                    server.accept()
                } catch (_: SocketException) {
                    break // server closed
                }
                thread(isDaemon = true) { handle(socket) }
            }
        }
    }

    @After
    fun tearDown() {
        server.close()
    }

    private fun handle(socket: Socket) = socket.use { s ->
        val input = s.getInputStream()
        val requestLine = readLine(input)
        val parts = requestLine.split(" ")
        method = parts.getOrNull(0)
        path = parts.getOrNull(1)
        val headers = mutableMapOf<String, String>()
        while (true) {
            val line = readLine(input)
            if (line.isEmpty()) break
            val idx = line.indexOf(':')
            if (idx > 0) headers[line.substring(0, idx).trim().lowercase()] = line.substring(idx + 1).trim()
        }
        contentType = headers["content-type"]
        authorization = headers["authorization"]
        val length = headers["content-length"]?.toInt() ?: 0
        val buf = ByteArray(length)
        var read = 0
        while (read < length) {
            val n = input.read(buf, read, length - read)
            if (n < 0) break
            read += n
        }
        body = buf
        if (replyDelayMs > 0) Thread.sleep(replyDelayMs)
        val bytes = replyBody.toByteArray(Charsets.UTF_8)
        val head = "HTTP/1.1 $replyStatus Status\r\n" +
            "Content-Type: application/json\r\n" +
            "Content-Length: ${bytes.size}\r\n" +
            "Connection: close\r\n\r\n"
        try {
            s.getOutputStream().apply {
                write(head.toByteArray(Charsets.US_ASCII))
                write(bytes)
                flush()
            }
        } catch (_: SocketException) {
            // Client gave up (timeout test) — nothing to do.
        }
    }

    /** Read one CRLF-terminated header line (without the CRLF). */
    private fun readLine(input: InputStream): String {
        val out = ByteArrayOutputStream()
        while (true) {
            val b = input.read()
            if (b < 0) break
            if (b == '\n'.code) break
            if (b != '\r'.code) out.write(b)
        }
        return out.toString(Charsets.ISO_8859_1.name())
    }

    private fun url(p: String = "/v1/audio/transcriptions") =
        "http://127.0.0.1:${server.localPort}$p"

    private fun engine(config: RemoteServerConfig, readTimeoutMs: Int = 5_000) =
        RemoteTranscriptionEngine(config, Dispatchers.IO, connectTimeoutMs = 5_000, readTimeoutMs = readTimeoutMs)

    @Test
    fun postsMultipartWavToTheExactUrlAndReturnsNormalizedText() = runBlocking {
        val samples = ShortArray(1600) { (it % 100).toShort() }
        val text = engine(RemoteServerConfig(url = url("/custom/path?x=1"), model = "large-v3"))
            .transcribeAudio(samples)

        assertEquals(" Hello world.", text)
        assertEquals("POST", method)
        assertEquals("/custom/path?x=1", path)
        assertNull(authorization)

        val ct = contentType!!
        assertTrue(ct, ct.startsWith("multipart/form-data; boundary="))
        val boundary = ct.substringAfter("boundary=")
        val bodyText = String(body, Charsets.ISO_8859_1)
        assertTrue(bodyText.contains("name=\"response_format\"\r\n\r\njson\r\n"))
        assertTrue(bodyText.contains("name=\"language\"\r\n\r\nen\r\n"))
        assertTrue(bodyText.contains("name=\"model\"\r\n\r\nlarge-v3\r\n"))
        assertTrue(bodyText.contains("name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF"))
        assertTrue(bodyText.endsWith("\r\n--$boundary--\r\n"))
        // Whole WAV made it through: 44-byte header + 2 bytes per sample.
        val wavStart = bodyText.indexOf("RIFF")
        val wavEnd = bodyText.lastIndexOf("\r\n--$boundary--")
        assertEquals(44 + samples.size * 2, wavEnd - wavStart)
    }

    @Test
    fun sendsBearerTokenOnlyWhenSet() = runBlocking {
        engine(RemoteServerConfig(url = url(), bearerToken = " secret ")).transcribeAudio(ShortArray(10))
        assertEquals("Bearer secret", authorization)

        engine(RemoteServerConfig(url = url(), bearerToken = "  ")).transcribeAudio(ShortArray(10))
        assertNull(authorization)
    }

    @Test
    fun omitsModelFieldWhenBlank() = runBlocking {
        engine(RemoteServerConfig(url = url(), model = "")).transcribeAudio(ShortArray(10))
        assertTrue(!String(body, Charsets.ISO_8859_1).contains("name=\"model\""))
    }

    @Test
    fun non2xxSurfacesServerErrorMessage() = runBlocking {
        replyStatus = 413
        replyBody = """{"error":{"message":"Audio too long"}}"""
        try {
            engine(RemoteServerConfig(url = url())).transcribeAudio(ShortArray(10))
            fail("expected failure")
        } catch (e: RemoteTranscriptionException) {
            assertEquals(413, e.httpStatus)
            assertEquals("HTTP 413: Audio too long", e.message)
        }
    }

    @Test
    fun readTimeoutIsReportedAsTimeout() = runBlocking {
        replyDelayMs = 1_000
        try {
            engine(RemoteServerConfig(url = url()), readTimeoutMs = 100).transcribeAudio(ShortArray(10))
            fail("expected failure")
        } catch (e: RemoteTranscriptionException) {
            assertNull(e.httpStatus)
            assertTrue(e.cause is SocketTimeoutException)
        }
    }

    @Test
    fun unreachableServerIsATransportError() = runBlocking {
        val deadUrl = url()
        server.close()
        try {
            engine(RemoteServerConfig(url = deadUrl)).transcribeAudio(ShortArray(10))
            fail("expected failure")
        } catch (e: RemoteTranscriptionException) {
            assertNull(e.httpStatus)
        }
    }
}
