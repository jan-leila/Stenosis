package dev.soupslurpr.transcribro.preferences

import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey

/** Preference pairs, the first is the preference key, and the second is the default value. */
data class PreferencesUiState(
    /** Pitch black background. */
    val pitchBlackBackground: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        (booleanPreferencesKey("PITCH_BLACK_BACKGROUND")),
        mutableStateOf(false)
    ),

    /**
     * Whether the user has accepted the privacy policy and license. The version suffix
     * is bumped whenever the policy text changes materially (V0.4.0: opt-in server
     * transcription can send audio off-device), which makes every existing user see
     * and re-accept the new text on next launch.
     */
    val acceptedPrivacyPolicyAndLicense: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        (booleanPreferencesKey("ACCEPTED_PRIVACY_POLICY_AND_LICENSE_V0.4.0")),
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

    // --- Per-feature streaming / noise suppression --------------------------------
    // The two transcription modes run in very different acoustic environments, so
    // every audio-pipeline knob is duplicated per mode rather than shared.

    /** Live preview during keyboard transcription. */
    val keyboardStreamingTranscription: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        booleanPreferencesKey("KEYBOARD_STREAMING_TRANSCRIPTION"),
        mutableStateOf(true),
    ),

    /** Live preview during conversation transcription. */
    val conversationStreamingTranscription: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        booleanPreferencesKey("CONVERSATION_STREAMING_TRANSCRIPTION"),
        mutableStateOf(true),
    ),

    /** RNNoise denoising applied to keyboard mic input. */
    val keyboardNoiseSuppressionEnabled: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        booleanPreferencesKey("KEYBOARD_NOISE_SUPPRESSION_ENABLED"),
        mutableStateOf(true),
    ),

    /** RNNoise denoising applied to conversation mic input. */
    val conversationNoiseSuppressionEnabled: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        booleanPreferencesKey("CONVERSATION_NOISE_SUPPRESSION_ENABLED"),
        mutableStateOf(true),
    ),

    // --- Per-feature VAD tunables -------------------------------------------------
    // SileroVadDetector takes start/end probability thresholds and a minimum silence
    // window. All three live behind the Advanced section because the failure modes
    // (cutting off mid-sentence, dropping quiet starts) need real testing to tune.

    /** Silero VAD probability ≥ this triggers segment start (keyboard). */
    val keyboardVadStartThreshold: Pair<Preferences.Key<Float>, MutableState<Float>> = Pair(
        floatPreferencesKey("KEYBOARD_VAD_START_THRESHOLD"),
        mutableStateOf(DEFAULT_VAD_START_THRESHOLD),
    ),

    /** Silero VAD probability falling below this starts the silence timer (keyboard). */
    val keyboardVadEndThreshold: Pair<Preferences.Key<Float>, MutableState<Float>> = Pair(
        floatPreferencesKey("KEYBOARD_VAD_END_THRESHOLD"),
        mutableStateOf(DEFAULT_VAD_END_THRESHOLD),
    ),

    /** Quiet duration (ms) after which the keyboard finalizes the current segment. */
    val keyboardMinSilenceMs: Pair<Preferences.Key<Int>, MutableState<Int>> = Pair(
        intPreferencesKey("KEYBOARD_MIN_SILENCE_MS"),
        mutableStateOf(DEFAULT_MIN_SILENCE_MS),
    ),

    /** Silero VAD probability ≥ this triggers segment start (conversation). */
    val conversationVadStartThreshold: Pair<Preferences.Key<Float>, MutableState<Float>> = Pair(
        floatPreferencesKey("CONVERSATION_VAD_START_THRESHOLD"),
        mutableStateOf(DEFAULT_VAD_START_THRESHOLD),
    ),

    /** Silero VAD probability falling below this starts the silence timer (conversation). */
    val conversationVadEndThreshold: Pair<Preferences.Key<Float>, MutableState<Float>> = Pair(
        floatPreferencesKey("CONVERSATION_VAD_END_THRESHOLD"),
        mutableStateOf(DEFAULT_VAD_END_THRESHOLD),
    ),

    /**
     * Quiet duration (ms) after which a conversation segment closes. Reused by the
     * diarization manager as the per-speaker silence window, since both knobs mean
     * the same thing to a user: "how long of quiet ends a turn".
     */
    val conversationMinSilenceMs: Pair<Preferences.Key<Int>, MutableState<Int>> = Pair(
        intPreferencesKey("CONVERSATION_MIN_SILENCE_MS"),
        mutableStateOf(DEFAULT_MIN_SILENCE_MS),
    ),

    /**
     * Cosine distance below which two embeddings are treated as the same speaker.
     * Tune up if speakers get over-split, down if they get over-merged. See
     * [dev.soupslurpr.transcribro.recognitionservice.diarization.SpeakerRegistry].
     */
    val conversationCosineThreshold: Pair<Preferences.Key<Float>, MutableState<Float>> = Pair(
        floatPreferencesKey("CONVERSATION_COSINE_THRESHOLD"),
        mutableStateOf(DEFAULT_COSINE_THRESHOLD),
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

    /** Active VAD model used by keyboard transcription. */
    val keyboardVadModelId: Pair<Preferences.Key<String>, MutableState<String?>> = Pair(
        stringPreferencesKey("KEYBOARD_VAD_MODEL_ID"),
        mutableStateOf("silero-vad-v5"),
    ),

    /** Active VAD model used by conversation transcription. */
    val conversationVadModelId: Pair<Preferences.Key<String>, MutableState<String?>> = Pair(
        stringPreferencesKey("CONVERSATION_VAD_MODEL_ID"),
        mutableStateOf("silero-vad-v5"),
    ),

    // --- Server transcription -------------------------------------------------------
    // The backend choice is per mode (like every other pipeline knob) so a user can,
    // say, dictate on-device in the keyboard but send long conversations to their home
    // server. The server connection itself is shared: there's one server, not one per
    // mode. Off by default — audio only leaves the device after an explicit opt-in.

    /** Send keyboard segment-final passes to the configured server instead of on-device whisper. */
    val keyboardServerTranscription: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        booleanPreferencesKey("KEYBOARD_SERVER_TRANSCRIPTION"),
        mutableStateOf(false),
    ),

    /** Send conversation segment-final passes to the configured server instead of on-device whisper. */
    val conversationServerTranscription: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        booleanPreferencesKey("CONVERSATION_SERVER_TRANSCRIPTION"),
        mutableStateOf(false),
    ),

    /**
     * Full URL of an OpenAI-compatible transcription endpoint, used verbatim (e.g.
     * `https://host/v1/audio/transcriptions`). Must be https; see
     * [dev.soupslurpr.transcribro.recognitionservice.remote.validateServerUrl].
     */
    val serverUrl: Pair<Preferences.Key<String>, MutableState<String>> = Pair(
        stringPreferencesKey("SERVER_URL"),
        mutableStateOf(""),
    ),

    /** Optional `model` form field. Blank means "let the server pick its default". */
    val serverModel: Pair<Preferences.Key<String>, MutableState<String>> = Pair(
        stringPreferencesKey("SERVER_MODEL"),
        mutableStateOf(""),
    ),

    /**
     * Optional bearer token. Stored in plain DataStore like every other preference —
     * no encrypted-storage dependency exists in the app and adding one for a single
     * optional field isn't worth it; the settings UI says so next to the field.
     */
    val serverToken: Pair<Preferences.Key<String>, MutableState<String>> = Pair(
        stringPreferencesKey("SERVER_TOKEN"),
        mutableStateOf(""),
    ),
) {
    companion object {
        const val DEFAULT_VAD_START_THRESHOLD: Float = 0.6f
        const val DEFAULT_VAD_END_THRESHOLD: Float = 0.45f
        const val DEFAULT_MIN_SILENCE_MS: Int = 3000
        const val DEFAULT_COSINE_THRESHOLD: Float = 0.35f
    }
}
