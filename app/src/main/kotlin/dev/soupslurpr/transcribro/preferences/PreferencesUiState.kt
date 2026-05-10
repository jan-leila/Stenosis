package dev.soupslurpr.transcribro.preferences

import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey

/** Preference pairs, the first is the preference key, and the second is the default value. */
data class PreferencesUiState(
    /** Pitch black background. */
    val pitchBlackBackground: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        (booleanPreferencesKey("PITCH_BLACK_BACKGROUND")),
        mutableStateOf(false)
    ),

    /** Whether the user has accepted the privacy policy and license. */
    val acceptedPrivacyPolicyAndLicense: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        (booleanPreferencesKey("ACCEPTED_PRIVACY_POLICY_AND_LICENSE_V0.3.0")),
        mutableStateOf(false)
    ),

    /** Whether to automatically switch to the previous input method when the keyboard is done transcribing */
    val autoSwitchToPreviousInputMethod: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        (booleanPreferencesKey("AUTO_SWITCH_TO_PREVIOUS_INPUT_METHOD")),
        mutableStateOf(false)
    ),

    /** Whether to automatically stop recognition when speech stops being detected. */
    val autoStopRecognition: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        (booleanPreferencesKey("AUTO_STOP_RECOGNITION")),
        mutableStateOf(false)
    ),

    /** Whether to automatically start recognition when switching from another input method. */
    val autoStartRecognition: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        (booleanPreferencesKey("AUTO_START_RECOGNITION")),
        mutableStateOf(true)
    ),

    /** Whether to automatically send transcription when speech stops being detected. */
    val autoSendTranscription: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        (booleanPreferencesKey("AUTO_SEND_TRANSCRIPTION")),
        mutableStateOf(false)
    ),

    /** Whether to show a live-updating preview of the current speech segment as it is being spoken. */
    val streamingTranscription: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        (booleanPreferencesKey("STREAMING_TRANSCRIPTION")),
        mutableStateOf(true)
    ),

    /**
     * Whether to run speaker diarization (pyannote + wespeaker) and label each
     * committed segment with a speaker id. Off by default — adds ~30MB of model RAM
     * and segments only commit after a per-speaker silence window, so the live UX
     * differs noticeably from the default VAD-driven flow.
     */
    val diarizationEnabled: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        (booleanPreferencesKey("DIARIZATION_ENABLED")),
        mutableStateOf(false)
    ),

    /**
     * Whether to apply RNNoise denoising to the mic input before transcription. On
     * by default — it noticeably improves whisper accuracy on noisy recordings and
     * helps the speaker-embedding model produce cleaner clusters. Adds ~10ms of
     * input latency and a small CPU cost (one 33-tap FIR resample plus the RNN per
     * 10ms frame).
     */
    val noiseSuppressionEnabled: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        (booleanPreferencesKey("NOISE_SUPPRESSION_ENABLED")),
        mutableStateOf(true)
    )
)