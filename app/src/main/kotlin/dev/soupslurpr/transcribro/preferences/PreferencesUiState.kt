package dev.soupslurpr.transcribro.preferences

import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey

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
    ),

    // --- Per-feature model selections ---------------------------------------------
    // Each holds a [ModelCatalog] entry id. Value of `null` means "no model picked
    // yet"; record paths refuse to start until the user installs and selects one.
    // VAD defaults to the single available silero entry so a freshly-installed-only
    // VAD doesn't require an extra picker step.

    /** Active whisper model used by the keyboard IME (final pass). */
    val keyboardWhisperModelId: Pair<Preferences.Key<String>, MutableState<String?>> = Pair(
        stringPreferencesKey("KEYBOARD_WHISPER_MODEL_ID"),
        mutableStateOf(null),
    ),

    /**
     * Optional separate whisper model for the keyboard's live preview pass. `null`
     * means "use the same model as the final pass" — the common case.
     */
    val keyboardWhisperStreamingModelId: Pair<Preferences.Key<String>, MutableState<String?>> = Pair(
        stringPreferencesKey("KEYBOARD_WHISPER_STREAMING_MODEL_ID"),
        mutableStateOf(null),
    ),

    /** Active whisper model used by in-app conversation transcription (final pass). */
    val conversationWhisperModelId: Pair<Preferences.Key<String>, MutableState<String?>> = Pair(
        stringPreferencesKey("CONVERSATION_WHISPER_MODEL_ID"),
        mutableStateOf(null),
    ),

    /**
     * Optional separate whisper model for the conversation screen's live preview pass.
     * `null` means "use the same model as the final pass".
     */
    val conversationWhisperStreamingModelId: Pair<Preferences.Key<String>, MutableState<String?>> = Pair(
        stringPreferencesKey("CONVERSATION_WHISPER_STREAMING_MODEL_ID"),
        mutableStateOf(null),
    ),

    /** Active pyannote-style speaker segmenter (conversation diarization only). */
    val conversationSegmenterModelId: Pair<Preferences.Key<String>, MutableState<String?>> = Pair(
        stringPreferencesKey("CONVERSATION_SEGMENTER_MODEL_ID"),
        mutableStateOf(null),
    ),

    /** Active speaker embedder (conversation diarization only). */
    val conversationEmbedderModelId: Pair<Preferences.Key<String>, MutableState<String?>> = Pair(
        stringPreferencesKey("CONVERSATION_EMBEDDER_MODEL_ID"),
        mutableStateOf(null),
    ),

    /** Active VAD model. Shared by both modes; only one VAD entry exists for now. */
    val vadModelId: Pair<Preferences.Key<String>, MutableState<String?>> = Pair(
        stringPreferencesKey("VAD_MODEL_ID"),
        mutableStateOf("silero-vad-v5"),
    ),
)