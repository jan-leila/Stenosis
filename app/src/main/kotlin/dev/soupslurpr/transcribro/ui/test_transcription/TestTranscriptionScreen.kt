package dev.soupslurpr.transcribro.ui.test_transcription

import android.Manifest
import android.content.ComponentName
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.runtime.rememberCoroutineScope
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import dev.soupslurpr.transcribro.recognitionservice.MainRecognitionService
import dev.soupslurpr.transcribro.recognitionservice.diarization.verifyDiarizationModels
import kotlinx.coroutines.launch

/**
 * Test transcription view. Bypasses the IME entirely and drives MainRecognitionService
 * directly — useful for trying transcription without enabling the keyboard.
 *
 * - Tap the button to start/stop continuous recognition.
 * - Segment-final partials get appended to the committed buffer.
 * - Streaming previews show in italic at the tail and are replaced when the segment-final lands.
 */
@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun TestTranscriptionScreen() {
    val context = LocalContext.current

    val micPermission = rememberPermissionState(Manifest.permission.RECORD_AUDIO)

    var recognizer by remember { mutableStateOf<SpeechRecognizer?>(null) }
    var isRecognizing by remember { mutableStateOf(false) }
    var committedBuffer by remember { mutableStateOf("") }
    var livePreview by remember { mutableStateOf("") }
    var status by remember { mutableStateOf("idle") }

    val coroutineScope = rememberCoroutineScope()
    var verifyReport by remember { mutableStateOf<String?>(null) }
    var verifyRunning by remember { mutableStateOf(false) }

    DisposableEffect(Unit) {
        val r = SpeechRecognizer.createSpeechRecognizer(
            context,
            ComponentName(context, MainRecognitionService::class.java)
        )
        r.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                isRecognizing = true
                status = "listening"
            }

            override fun onBeginningOfSpeech() {
                status = "speaking"
            }

            override fun onRmsChanged(rmsdB: Float) {}

            override fun onBufferReceived(buffer: ByteArray?) {}

            override fun onEndOfSpeech() {
                status = "processing"
            }

            override fun onError(error: Int) {
                isRecognizing = false
                livePreview = ""
                status = "error: $error"
            }

            override fun onResults(results: Bundle?) {
                // The service emits each segment as a non-streaming partial which we already
                // appended to committedBuffer. onResults arrives at the end with the FULL
                // joined transcript — so replace, don't append (otherwise we double-print).
                val finalText = results
                    ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.firstOrNull()
                    ?.trim()
                    .orEmpty()
                if (finalText.isNotBlank()) {
                    committedBuffer = finalText
                }
                livePreview = ""
                isRecognizing = false
                status = "idle"
            }

            override fun onPartialResults(partialResults: Bundle?) {
                if (partialResults == null) return
                val isStreamingPreview = partialResults.getBoolean(
                    MainRecognitionService.RESULTS_STREAMING_PARTIAL,
                    false
                )
                val text = partialResults
                    .getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.firstOrNull()
                    .orEmpty()
                if (isStreamingPreview) {
                    livePreview = text.trim()
                } else {
                    if (text.isNotBlank()) {
                        committedBuffer = appendSegment(committedBuffer, text)
                    }
                    livePreview = ""
                }
            }

            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
        recognizer = r
        onDispose {
            r.destroy()
            recognizer = null
        }
    }

    LaunchedEffect(Unit) {
        if (!micPermission.status.isGranted) {
            micPermission.launchPermissionRequest()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
            Button(
                onClick = {
                    val r = recognizer ?: return@Button
                    if (isRecognizing) {
                        r.stopListening()
                    } else {
                        committedBuffer = ""
                        livePreview = ""
                        r.startListening(
                            Intent().apply {
                                putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                                // Long-form recognition: don't auto-stop on first VAD end.
                                putExtra(MainRecognitionService.EXTRA_AUTO_STOP, false)
                                putExtra(MainRecognitionService.EXTRA_STREAMING, true)
                            }
                        )
                    }
                },
                enabled = micPermission.status.isGranted && recognizer != null,
            ) {
                Text(if (isRecognizing) "Stop transcription" else "Start transcription")
            }
            Spacer(Modifier.size(12.dp))
            OutlinedButton(
                onClick = {
                    committedBuffer = ""
                    livePreview = ""
                },
                enabled = !isRecognizing,
            ) {
                Text("Clear")
            }
        }
        Spacer(Modifier.size(8.dp))
        Text(
            text = "status: $status",
            style = MaterialTheme.typography.labelMedium,
        )
        Spacer(Modifier.size(12.dp))
        ElevatedCard(
            modifier = Modifier
                .fillMaxWidth()
                // Take whatever vertical space is left after the buttons + verify panel,
                // so the transcript scrolls inside the card instead of pushing siblings
                // below the screen.
                .weight(1f),
        ) {
            val scrollState = rememberScrollState()
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp)
                    .verticalScroll(scrollState),
                verticalArrangement = Arrangement.Top,
            ) {
                Text(
                    text = renderTranscript(committedBuffer, livePreview),
                    style = MaterialTheme.typography.bodyLarge,
                )
            }
        }
        Spacer(Modifier.size(12.dp))
        Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
            OutlinedButton(
                onClick = {
                    if (verifyRunning) return@OutlinedButton
                    verifyRunning = true
                    verifyReport = "running…"
                    coroutineScope.launch {
                        verifyReport = try {
                            verifyDiarizationModels(context)
                        } catch (t: Throwable) {
                            "verify failed: ${t::class.simpleName}: ${t.message}"
                        } finally {
                            verifyRunning = false
                        }
                    }
                },
                enabled = !verifyRunning,
            ) {
                Text(if (verifyRunning) "Verifying…" else "Verify pyannote models")
            }
        }
        verifyReport?.let { report ->
            Spacer(Modifier.size(8.dp))
            ElevatedCard(modifier = Modifier.fillMaxWidth()) {
                val scrollState = rememberScrollState()
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(12.dp)
                        .verticalScroll(scrollState),
                ) {
                    Text(
                        text = report,
                        style = MaterialTheme.typography.bodySmall.copy(
                            fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace,
                        ),
                    )
                }
            }
        }
    }
}

private fun appendSegment(existing: String, segment: String): String {
    val trimmed = segment.trim()
    if (trimmed.isEmpty()) return existing
    return if (existing.isEmpty()) trimmed else "$existing $trimmed"
}

@Composable
private fun renderTranscript(committed: String, preview: String): AnnotatedString =
    buildAnnotatedString {
        append(committed)
        if (preview.isNotEmpty()) {
            if (committed.isNotEmpty()) append(" ")
            withStyle(
                SpanStyle(
                    color = MaterialTheme.colorScheme.primary,
                    fontStyle = FontStyle.Italic,
                )
            ) {
                append(preview)
            }
        }
        if (committed.isEmpty() && preview.isEmpty()) {
            withStyle(SpanStyle(color = MaterialTheme.colorScheme.onSurfaceVariant)) {
                append("(transcription will appear here)")
            }
        }
    }
