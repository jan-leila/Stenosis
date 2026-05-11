package dev.soupslurpr.transcribro.ui.settings

import android.app.Application
import androidx.datastore.preferences.core.edit
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import dev.soupslurpr.transcribro.dataStore
import dev.soupslurpr.transcribro.models.InstallProgress
import dev.soupslurpr.transcribro.models.InstalledModelsRepository
import dev.soupslurpr.transcribro.models.ModelCatalog
import dev.soupslurpr.transcribro.models.ModelEntry
import dev.soupslurpr.transcribro.models.ModelInstaller
import dev.soupslurpr.transcribro.models.ModelKind
import dev.soupslurpr.transcribro.preferences.PreferencesUiState
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

/**
 * UI state per catalog entry. Either [Idle] (installed or not), [Downloading] with
 * a 0..1 fraction (null total = indeterminate), or [Failed] holding the last error
 * message until the user retries or dismisses.
 */
sealed class ModelUiState {
    data class Idle(val installed: Boolean) : ModelUiState()
    data class Downloading(
        val bytesDownloaded: Long,
        val totalBytes: Long?,
        val phase: Phase,
    ) : ModelUiState() {
        enum class Phase { Connecting, Downloading, Verifying }
        val fraction: Float?
            get() = if (totalBytes != null && totalBytes > 0L) {
                (bytesDownloaded.toFloat() / totalBytes.toFloat()).coerceIn(0f, 1f)
            } else null
    }

    data class Failed(val message: String) : ModelUiState()
}

class ModelsViewModel(application: Application) : AndroidViewModel(application) {
    private val installed = InstalledModelsRepository(application)
    private val installer = ModelInstaller(installed)
    private val dataStore = application.dataStore

    /**
     * For each [ModelKind], the list of feature-selection prefs that should auto-adopt
     * a freshly-installed model when they're currently unset. Streaming-preview prefs
     * are intentionally absent — their `null` value means "reuse the final-pass model",
     * so auto-promoting would change semantics.
     */
    private val autoPromoteKeys: Map<ModelKind, List<androidx.datastore.preferences.core.Preferences.Key<String>>> =
        run {
            val defaults = PreferencesUiState()
            mapOf(
                ModelKind.WhisperTranscriber to listOf(
                    defaults.keyboardWhisperModelId.first,
                    defaults.conversationWhisperModelId.first,
                ),
                ModelKind.VoiceActivityDetector to listOf(defaults.vadModelId.first),
                ModelKind.SpeakerSegmenter to listOf(defaults.conversationSegmenterModelId.first),
                ModelKind.SpeakerEmbedder to listOf(defaults.conversationEmbedderModelId.first),
            )
        }

    private val _statesById = MutableStateFlow<Map<String, ModelUiState>>(initialStates())
    val statesById: StateFlow<Map<String, ModelUiState>> = _statesById.asStateFlow()

    val installedState: StateFlow<dev.soupslurpr.transcribro.models.InstalledModelsState> =
        installed.state

    val entries: List<ModelEntry> get() = ModelCatalog.entries

    /** Active install jobs, so cancel can interrupt cooperatively. */
    private val installJobs = mutableMapOf<String, Job>()

    init {
        // Backfill: if the user already has models installed but the corresponding
        // feature prefs are still null (e.g. they installed under an older build that
        // didn't auto-promote), pick the first installed model of each kind.
        viewModelScope.launch {
            for ((kind, _) in autoPromoteKeys) {
                val firstInstalled = ModelCatalog.forKind(kind)
                    .firstOrNull { installed.isInstalled(it) }
                    ?: continue
                promoteAsDefault(firstInstalled)
            }
        }
    }

    private fun initialStates(): Map<String, ModelUiState> =
        ModelCatalog.entries.associate { entry ->
            entry.id to ModelUiState.Idle(installed.isInstalled(entry))
        }

    fun install(entry: ModelEntry) {
        if (installJobs[entry.id]?.isActive == true) return
        installJobs[entry.id] = viewModelScope.launch {
            installer.install(entry).collect { progress ->
                _statesById.update(entry.id) { mapProgress(progress) }
                if (progress is InstallProgress.Done) promoteAsDefault(entry)
            }
        }
    }

    /**
     * For every feature-selection pref of this entry's kind, set it to [entry] if it's
     * not already set. Lets a fresh install land as the active model for keyboard +
     * conversation transcription (etc.) without the user having to revisit settings.
     */
    private suspend fun promoteAsDefault(entry: ModelEntry) {
        val keys = autoPromoteKeys[entry.kind] ?: return
        dataStore.edit { prefs ->
            for (key in keys) {
                if (prefs[key] == null) prefs[key] = entry.id
            }
        }
    }

    fun cancel(entry: ModelEntry) {
        installJobs[entry.id]?.cancel()
        installJobs.remove(entry.id)
        _statesById.update(entry.id) { ModelUiState.Idle(installed.isInstalled(entry)) }
    }

    fun uninstall(entry: ModelEntry) {
        installed.delete(entry)
        _statesById.update(entry.id) { ModelUiState.Idle(installed.isInstalled(entry)) }
    }

    fun dismissFailure(entry: ModelEntry) {
        _statesById.update(entry.id) { ModelUiState.Idle(installed.isInstalled(entry)) }
    }

    private fun mapProgress(progress: InstallProgress): ModelUiState = when (progress) {
        is InstallProgress.Connecting ->
            ModelUiState.Downloading(0L, null, ModelUiState.Downloading.Phase.Connecting)
        is InstallProgress.Downloading ->
            ModelUiState.Downloading(progress.bytesDownloaded, progress.totalBytes, ModelUiState.Downloading.Phase.Downloading)
        is InstallProgress.Verifying ->
            ModelUiState.Downloading(0L, null, ModelUiState.Downloading.Phase.Verifying)
        is InstallProgress.Done -> ModelUiState.Idle(installed = true)
        is InstallProgress.Failed -> ModelUiState.Failed(progress.message)
    }

    private fun MutableStateFlow<Map<String, ModelUiState>>.update(
        key: String,
        compute: () -> ModelUiState,
    ) {
        value = value.toMutableMap().apply { this[key] = compute() }
    }

    class Factory(private val application: Application) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T =
            ModelsViewModel(application) as T
    }
}
