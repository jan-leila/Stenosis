package dev.soupslurpr.transcribro.preferences

import androidx.compose.runtime.mutableStateOf
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
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
     * This function is only called from this ViewModel's init
     */
    private suspend fun populatePreferencesFromDatastore() {
        dataStore.data.map { preferences ->
            _uiState.update { currentState ->
                currentState.copy(
                    pitchBlackBackground = Pair(
                        uiState.value.pitchBlackBackground.first,
                        mutableStateOf(
                            preferences[uiState.value.pitchBlackBackground.first] ?: uiState.value
                                .pitchBlackBackground.second.value
                        )
                    ),
                    acceptedPrivacyPolicyAndLicense = Pair(
                        uiState.value.acceptedPrivacyPolicyAndLicense.first,
                        mutableStateOf(
                            preferences[uiState.value.acceptedPrivacyPolicyAndLicense.first] ?: uiState.value
                                .acceptedPrivacyPolicyAndLicense.second.value
                        )
                    ),
                    autoSwitchToPreviousInputMethod = Pair(
                        uiState.value.autoSwitchToPreviousInputMethod.first,
                        mutableStateOf(
                            preferences[uiState.value.autoSwitchToPreviousInputMethod
                                .first] ?: uiState.value
                                .autoSwitchToPreviousInputMethod.second.value
                        )
                    ),
                    autoStopRecognition = Pair(
                        uiState.value.autoStopRecognition.first,
                        mutableStateOf(
                            preferences[uiState.value.autoStopRecognition
                                .first] ?: uiState.value
                                .autoStopRecognition.second.value
                        )
                    ),
                    autoStartRecognition = Pair(
                        uiState.value.autoStartRecognition.first,
                        mutableStateOf(
                            preferences[uiState.value.autoStartRecognition
                                .first] ?: uiState.value
                                .autoStartRecognition.second.value
                        )
                    ),
                    autoSendTranscription = Pair(
                        uiState.value.autoSendTranscription.first,
                        mutableStateOf(
                            preferences[uiState.value.autoSendTranscription
                                .first] ?: uiState.value
                                .autoSendTranscription.second.value
                        )
                    ),
                    streamingTranscription = Pair(
                        uiState.value.streamingTranscription.first,
                        mutableStateOf(
                            preferences[uiState.value.streamingTranscription
                                .first] ?: uiState.value
                                .streamingTranscription.second.value
                        )
                    ),
                    diarizationEnabled = Pair(
                        uiState.value.diarizationEnabled.first,
                        mutableStateOf(
                            preferences[uiState.value.diarizationEnabled
                                .first] ?: uiState.value
                                .diarizationEnabled.second.value
                        )
                    ),
                    noiseSuppressionEnabled = Pair(
                        uiState.value.noiseSuppressionEnabled.first,
                        mutableStateOf(
                            preferences[uiState.value.noiseSuppressionEnabled
                                .first] ?: uiState.value
                                .noiseSuppressionEnabled.second.value
                        )
                    ),
                    keyboardWhisperModelId = Pair(
                        uiState.value.keyboardWhisperModelId.first,
                        mutableStateOf(
                            preferences[uiState.value.keyboardWhisperModelId.first]
                                ?: uiState.value.keyboardWhisperModelId.second.value
                        )
                    ),
                    keyboardWhisperStreamingModelId = Pair(
                        uiState.value.keyboardWhisperStreamingModelId.first,
                        mutableStateOf(
                            preferences[uiState.value.keyboardWhisperStreamingModelId.first]
                                ?: uiState.value.keyboardWhisperStreamingModelId.second.value
                        )
                    ),
                    conversationWhisperModelId = Pair(
                        uiState.value.conversationWhisperModelId.first,
                        mutableStateOf(
                            preferences[uiState.value.conversationWhisperModelId.first]
                                ?: uiState.value.conversationWhisperModelId.second.value
                        )
                    ),
                    conversationWhisperStreamingModelId = Pair(
                        uiState.value.conversationWhisperStreamingModelId.first,
                        mutableStateOf(
                            preferences[uiState.value.conversationWhisperStreamingModelId.first]
                                ?: uiState.value.conversationWhisperStreamingModelId.second.value
                        )
                    ),
                    conversationSegmenterModelId = Pair(
                        uiState.value.conversationSegmenterModelId.first,
                        mutableStateOf(
                            preferences[uiState.value.conversationSegmenterModelId.first]
                                ?: uiState.value.conversationSegmenterModelId.second.value
                        )
                    ),
                    conversationEmbedderModelId = Pair(
                        uiState.value.conversationEmbedderModelId.first,
                        mutableStateOf(
                            preferences[uiState.value.conversationEmbedderModelId.first]
                                ?: uiState.value.conversationEmbedderModelId.second.value
                        )
                    ),
                    vadModelId = Pair(
                        uiState.value.vadModelId.first,
                        mutableStateOf(
                            preferences[uiState.value.vadModelId.first]
                                ?: uiState.value.vadModelId.second.value
                        )
                    ),
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

    class PreferencesViewModelFactory(private val dataStore: DataStore<Preferences>) : ViewModelProvider.Factory {
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            if (modelClass.isAssignableFrom(PreferencesViewModel::class.java)) {
                @Suppress("UNCHECKED_CAST")
                return PreferencesViewModel(dataStore) as T
            }
            throw IllegalArgumentException("Unknown ViewModel class $modelClass")
        }
    }
}