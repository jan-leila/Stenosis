package dev.soupslurpr.transcribro.recognitionservice

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.fail
import org.junit.Test
import java.io.IOException

class FallbackTranscriptionEngineTest {
    private class Fixed(val text: String) : TranscriptionEngine {
        var calls = 0
        override suspend fun transcribeAudio(data: ShortArray): String {
            calls++
            return text
        }
    }

    private class Throwing(val error: Exception) : TranscriptionEngine {
        override suspend fun transcribeAudio(data: ShortArray): String = throw error
    }

    @Test
    fun usesPrimaryWhenItSucceeds() = runBlocking {
        val fallback = Fixed(" local")
        var reported: Throwable? = null
        val engine = FallbackTranscriptionEngine(Fixed(" remote"), fallback) { reported = it }
        assertEquals(" remote", engine.transcribeAudio(ShortArray(1)))
        assertEquals(0, fallback.calls)
        assertNull(reported)
    }

    @Test
    fun fallsBackAndReportsWhenPrimaryFails() = runBlocking {
        val error = IOException("down")
        var reported: Throwable? = null
        val engine = FallbackTranscriptionEngine(Throwing(error), Fixed(" local")) { reported = it }
        assertEquals(" local", engine.transcribeAudio(ShortArray(1)))
        assertSame(error, reported)
    }

    @Test
    fun rethrowsWhenNoFallback() = runBlocking {
        val error = IOException("down")
        val engine = FallbackTranscriptionEngine(Throwing(error), fallback = null)
        try {
            engine.transcribeAudio(ShortArray(1))
            fail("expected failure")
        } catch (e: IOException) {
            assertSame(error, e)
        }
    }

    @Test
    fun cancellationIsNotTreatedAsFailure() = runBlocking {
        val fallback = Fixed(" local")
        val engine = FallbackTranscriptionEngine(Throwing(CancellationException("cancelled")), fallback)
        try {
            engine.transcribeAudio(ShortArray(1))
            fail("expected cancellation")
        } catch (_: CancellationException) {
        }
        assertEquals(0, fallback.calls)
    }
}
