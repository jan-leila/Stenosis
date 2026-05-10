package dev.soupslurpr.transcribro.ui.transcribe

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
import androidx.compose.runtime.collectAsState
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
import androidx.lifecycle.viewmodel.compose.viewModel
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import dev.soupslurpr.transcribro.dataStore
import dev.soupslurpr.transcribro.preferences.PreferencesViewModel
import dev.soupslurpr.transcribro.recognitionservice.MainRecognitionService
import dev.soupslurpr.transcribro.recognitionservice.diarization.SpeakerLabelStore

/**
 * Transcription view. Drives [MainRecognitionService] directly without going through
 * the IME — useful for trying transcription, dictation-style note-taking, or any case
 * where you don't want voice input bound to a text field.
 *
 * - Tap the start/stop button to begin / end recognition.
 * - In default VAD mode, segment-final partials get appended inline; streaming
 *   previews show in italic at the tail.
 * - In diarization mode (enabled in Settings), each segment-final partial carries a
 *   speaker id; the screen looks the display name up via [SpeakerLabelStore] and
 *   renders one line per speaker turn.
 */
@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun TranscribeScreen() {
    val context = LocalContext.current

    val micPermission = rememberPermissionState(Manifest.permission.RECORD_AUDIO)

    var recognizer by remember { mutableStateOf<SpeechRecognizer?>(null) }
    var isRecognizing by remember { mutableStateOf(false) }
    var committedBuffer by remember { mutableStateOf("") }
    var livePreview by remember { mutableStateOf("") }
    var status by remember { mutableStateOf("idle") }

    val preferencesViewModel: PreferencesViewModel = viewModel(
        factory = PreferencesViewModel.PreferencesViewModelFactory(context.dataStore),
    )
    val preferencesUiState by preferencesViewModel.uiState.collectAsState()
    val diarizationEnabled = preferencesUiState.diarizationEnabled.second.value

    // Per-screen label store. When we wire renaming later (e.g. LLM-driven speaker
    // naming after a session), this is the surface that pulls names from. For now it
    // just gives us "Speaker 1" / "Speaker 2" defaults.
    val speakerLabelStore = remember { SpeakerLabelStore() }

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
                // The service emits each segment as a non-streaming partial which we
                // already appended to committedBuffer. onResults arrives at the end with
                // the FULL joined transcript — replace rather than append (otherwise we
                // double-print).
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
                    return
                }

                if (text.isBlank()) {
                    livePreview = ""
                    return
                }

                // Segment-final partial. In diarization mode the bundle also carries a
                // speaker id; we look up the display name via the label store and prefix
                // the line. In VAD mode there's no speaker id and we append inline.
                val speakerId = if (partialResults.containsKey(MainRecognitionService.RESULTS_SPEAKER_ID)) {
                    partialResults.getInt(MainRecognitionService.RESULTS_SPEAKER_ID)
                } else null

                committedBuffer = if (speakerId != null) {
                    val label = speakerLabelStore.displayName(speakerId)
                    appendSpeakerLine(committedBuffer, label, text)
                } else {
                    appendSegment(committedBuffer, text)
                }
                livePreview = ""
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
                                putExtra(MainRecognitionService.EXTRA_DIARIZATION, diarizationEnabled)
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
                // Bound the transcript card so it scrolls inside this height rather than
                // pushing the buttons off the bottom of the screen.
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
    }
}

private fun appendSegment(existing: String, segment: String): String {
    val trimmed = segment.trim()
    if (trimmed.isEmpty()) return existing
    return if (existing.isEmpty()) trimmed else "$existing $trimmed"
}

private fun appendSpeakerLine(existing: String, speakerLabel: String, segment: String): String {
    val trimmed = segment.trim()
    if (trimmed.isEmpty()) return existing
    val line = "$speakerLabel: $trimmed"
    return if (existing.isEmpty()) line else "$existing\n$line"
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
