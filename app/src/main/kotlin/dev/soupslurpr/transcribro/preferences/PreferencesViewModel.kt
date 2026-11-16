package dev.soupslurpr.transcribro.preferences

import androidx.compose.runtime.mutableStateOf
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class PreferencesViewModel(private val dataStore: DataStore<Preferences>) : ViewModel() {
    /**
     * Settings state
     */
    private val _uiState = MutableStateFlow(PreferencesUiState())
    val uiState: StateFlow<PreferencesUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            populatePreferencesFromDatastore()
        }
    }

    /**
     * Populate the values of the preferences from the Preferences DataStore.
     * This function is only called from this ViewModel's init.
     *
     * Migration: streaming / noise-suppression / VAD-model used to live under
     * single shared keys ([LEGACY_STREAMING_KEY], [LEGACY_NOISE_KEY], [LEGACY_VAD_KEY]).
     * When the per-feature key isn't set we fall back to the old shared value so
     * existing users don't get reset to defaults on the first launch after the split.
     */
    private suspend fun populatePreferencesFromDatastore() {
        dataStore.data.map { preferences ->
            val legacyStreaming = preferences[LEGACY_STREAMING_KEY]
            val legacyNoise = preferences[LEGACY_NOISE_KEY]
            val legacyVad = preferences[LEGACY_VAD_KEY]
            _uiState.update { currentState ->
                currentState.copy(
                    pitchBlackBackground = bool(currentState.pitchBlackBackground, preferences),
                    acceptedPrivacyPolicyAndLicense = bool(
                        currentState.acceptedPrivacyPolicyAndLicense, preferences
                    ),
                    autoSwitchToPreviousInputMethod = bool(
                        currentState.autoSwitchToPreviousInputMethod, preferences
                    ),
                    autoStopRecognition = bool(currentState.autoStopRecognition, preferences),
                    autoStartRecognition = bool(currentState.autoStartRecognition, preferences),
                    autoSendTranscription = bool(currentState.autoSendTranscription, preferences),
                    diarizationEnabled = bool(currentState.diarizationEnabled, preferences),
                    keyboardStreamingTranscription = bool(
                        currentState.keyboardStreamingTranscription, preferences, legacyStreaming
                    ),
                    conversationStreamingTranscription = bool(
                        currentState.conversationStreamingTranscription, preferences, legacyStreaming
                    ),
                    keyboardNoiseSuppressionEnabled = bool(
                        currentState.keyboardNoiseSuppressionEnabled, preferences, legacyNoise
                    ),
                    conversationNoiseSuppressionEnabled = bool(
                        currentState.conversationNoiseSuppressionEnabled, preferences, legacyNoise
                    ),
                    keyboardVadStartThreshold = float(
                        currentState.keyboardVadStartThreshold, preferences
                    ),
                    keyboardVadEndThreshold = float(
                        currentState.keyboardVadEndThreshold, preferences
                    ),
                    keyboardMinSilenceMs = int(currentState.keyboardMinSilenceMs, preferences),
                    conversationVadStartThreshold = float(
                        currentState.conversationVadStartThreshold, preferences
                    ),
                    conversationVadEndThreshold = float(
                        currentState.conversationVadEndThreshold, preferences
                    ),
                    conversationMinSilenceMs = int(
                        currentState.conversationMinSilenceMs, preferences
                    ),
                    conversationCosineThreshold = float(
                        currentState.conversationCosineThreshold, preferences
                    ),
                    keyboardWhisperModelId = nullableString(
                        currentState.keyboardWhisperModelId, preferences
                    ),
                    keyboardWhisperStreamingModelId = nullableString(
                        currentState.keyboardWhisperStreamingModelId, preferences
                    ),
                    conversationWhisperModelId = nullableString(
                        currentState.conversationWhisperModelId, preferences
                    ),
                    conversationWhisperStreamingModelId = nullableString(
                        currentState.conversationWhisperStreamingModelId, preferences
                    ),
                    conversationSegmenterModelId = nullableString(
                        currentState.conversationSegmenterModelId, preferences
                    ),
                    conversationEmbedderModelId = nullableString(
                        currentState.conversationEmbedderModelId, preferences
                    ),
                    keyboardVadModelId = nullableString(
                        currentState.keyboardVadModelId, preferences, legacyVad
                    ),
                    conversationVadModelId = nullableString(
                        currentState.conversationVadModelId, preferences, legacyVad
                    ),
                    keyboardServerTranscription = bool(
                        currentState.keyboardServerTranscription, preferences
                    ),
                    conversationServerTranscription = bool(
                        currentState.conversationServerTranscription, preferences
                    ),
                    serverUrl = string(currentState.serverUrl, preferences),
                    serverModel = string(currentState.serverModel, preferences),
                    serverToken = string(currentState.serverToken, preferences),
                )
            }
        }.collect()
    }

    /**
     * Set a preference to a value and save to Preferences DataStore
     */
    fun setPreference(key: Preferences.Key<Boolean>, value: Boolean) {
        viewModelScope.launch {
            dataStore.edit { preferences ->
                preferences[key] = value
            }
        }
    }

    /**
     * Set a string preference. Passing `null` removes the key, which the populate
     * path interprets as "fall back to the in-memory default".
     */
    @JvmName("setStringPreference")
    fun setPreference(key: Preferences.Key<String>, value: String?) {
        viewModelScope.launch {
            dataStore.edit { preferences ->
                if (value == null) preferences.remove(key) else preferences[key] = value
            }
        }
    }

    @JvmName("setIntPreference")
    fun setPreference(key: Preferences.Key<Int>, value: Int) {
        viewModelScope.launch {
            dataStore.edit { preferences ->
                preferences[key] = value
            }
        }
    }

    @JvmName("setFloatPreference")
    fun setPreference(key: Preferences.Key<Float>, value: Float) {
        viewModelScope.launch {
            dataStore.edit { preferences ->
                preferences[key] = value
            }
        }
    }

    /** Reset a stored preference to its compile-time default. */
    @JvmName("resetIntPreference")
    fun resetPreference(key: Preferences.Key<Int>) {
        viewModelScope.launch {
            dataStore.edit { preferences -> preferences.remove(key) }
        }
    }

    @JvmName("resetFloatPreference")
    fun resetPreference(key: Preferences.Key<Float>) {
        viewModelScope.launch {
            dataStore.edit { preferences -> preferences.remove(key) }
        }
    }

    class PreferencesViewModelFactory(private val dataStore: DataStore<Preferences>) : ViewModelProvider.Factory {
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            if (modelClass.isAssignableFrom(PreferencesViewModel::class.java)) {
                @Suppress("UNCHECKED_CAST")
                return PreferencesViewModel(dataStore) as T
            }
            throw IllegalArgumentException("Unknown ViewModel class $modelClass")
        }
    }

    companion object {
        // Old shared preference keys, retained only so populate-time migration can
        // read their values. Nothing writes to them anymore; the per-feature keys
        // own the canonical state. Defaults match the originals (true / true / "silero-vad-v5").
        private val LEGACY_STREAMING_KEY = booleanPreferencesKey("STREAMING_TRANSCRIPTION")
        private val LEGACY_NOISE_KEY = booleanPreferencesKey("NOISE_SUPPRESSION_ENABLED")
        private val LEGACY_VAD_KEY = stringPreferencesKey("VAD_MODEL_ID")

        private fun bool(
            pair: Pair<Preferences.Key<Boolean>, androidx.compose.runtime.MutableState<Boolean>>,
            preferences: Preferences,
            fallback: Boolean? = null,
        ): Pair<Preferences.Key<Boolean>, androidx.compose.runtime.MutableState<Boolean>> = Pair(
            pair.first,
            mutableStateOf(preferences[pair.first] ?: fallback ?: pair.second.value),
        )

        private fun int(
            pair: Pair<Preferences.Key<Int>, androidx.compose.runtime.MutableState<Int>>,
            preferences: Preferences,
        ): Pair<Preferences.Key<Int>, androidx.compose.runtime.MutableState<Int>> = Pair(
            pair.first,
            mutableStateOf(preferences[pair.first] ?: pair.second.value),
        )

        private fun float(
            pair: Pair<Preferences.Key<Float>, androidx.compose.runtime.MutableState<Float>>,
            preferences: Preferences,
        ): Pair<Preferences.Key<Float>, androidx.compose.runtime.MutableState<Float>> = Pair(
            pair.first,
            mutableStateOf(preferences[pair.first] ?: pair.second.value),
        )

        private fun string(
            pair: Pair<Preferences.Key<String>, androidx.compose.runtime.MutableState<String>>,
            preferences: Preferences,
        ): Pair<Preferences.Key<String>, androidx.compose.runtime.MutableState<String>> = Pair(
            pair.first,
            mutableStateOf(preferences[pair.first] ?: pair.second.value),
        )

        private fun nullableString(
            pair: Pair<Preferences.Key<String>, androidx.compose.runtime.MutableState<String?>>,
            preferences: Preferences,
            fallback: String? = null,
        ): Pair<Preferences.Key<String>, androidx.compose.runtime.MutableState<String?>> = Pair(
            pair.first,
            mutableStateOf(preferences[pair.first] ?: fallback ?: pair.second.value),
        )
    }
}
