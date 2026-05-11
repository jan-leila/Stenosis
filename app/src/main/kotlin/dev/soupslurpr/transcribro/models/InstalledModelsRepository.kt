package dev.soupslurpr.transcribro.models

import android.content.Context
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File

/**
 * Knows which models are present on disk.
 *
 * On-disk layout under `<filesDir>/models/`:
 * ```
 * <kind>/<entryId>/model.bin       # the verified, fully-downloaded file
 * <kind>/<entryId>/model.bin.part  # in-flight download — never trusted as installed
 * <kind>/<entryId>/sha256          # the verified sha256 (lowercase hex). Presence of
 *                                    this file is the install marker. Absent → either
 *                                    never installed or a previous install was aborted.
 * ```
 *
 * The `sha256` sidecar lets us distinguish "fully verified + atomically renamed" from
 * "raw bytes happen to exist". A crash mid-download leaves only `model.bin.part` and
 * no sidecar, so the next install can resume from scratch without confusing this
 * repository.
 *
 * The state Flow is updated on every install/uninstall via [refresh]. Callers that want
 * a snapshot can read [state].value directly.
 */
class InstalledModelsRepository(context: Context) {
    private val rootDir: File = File(context.filesDir, "models")

    private val _state = MutableStateFlow(InstalledModelsState.EMPTY)
    val state: StateFlow<InstalledModelsState> = _state.asStateFlow()

    init {
        rootDir.mkdirs()
        refresh()
    }

    /** Re-scan the filesystem and update [state]. Safe to call from any thread. */
    fun refresh() {
        val installed = mutableMapOf<String, InstalledModel>()
        for (entry in ModelCatalog.entries) {
            val dir = dirFor(entry)
            val file = modelFileIn(dir)
            val shaSidecar = shaSidecarIn(dir)
            if (file.exists() && file.length() > 0L && shaSidecar.exists()) {
                installed[entry.id] = InstalledModel(
                    entry = entry,
                    file = file,
                    sizeBytes = file.length(),
                )
            }
        }
        _state.value = InstalledModelsState(byId = installed)
        Log.i(TAG, "refresh: ${installed.size} installed model(s)")
    }

    fun isInstalled(entry: ModelEntry): Boolean = _state.value.byId.containsKey(entry.id)

    fun fileFor(entry: ModelEntry): File? = _state.value.byId[entry.id]?.file

    fun delete(entry: ModelEntry): Boolean {
        val dir = dirFor(entry)
        val ok = dir.deleteRecursively()
        refresh()
        return ok
    }

    /** Path layout helpers — also used by [ModelInstaller]. */
    internal fun dirFor(entry: ModelEntry): File =
        File(rootDir, "${entry.kind.name}/${entry.id}")

    internal fun modelFileIn(dir: File): File = File(dir, "model.bin")
    internal fun partFileIn(dir: File): File = File(dir, "model.bin.part")
    internal fun shaSidecarIn(dir: File): File = File(dir, "sha256")

    companion object {
        private const val TAG = "InstalledModels"
    }
}

data class InstalledModel(
    val entry: ModelEntry,
    val file: File,
    val sizeBytes: Long,
)

data class InstalledModelsState(val byId: Map<String, InstalledModel>) {
    val totalSizeBytes: Long get() = byId.values.sumOf { it.sizeBytes }

    companion object {
        val EMPTY = InstalledModelsState(byId = emptyMap())
    }
}
