package dev.soupslurpr.transcribro.ui.settings

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.soupslurpr.transcribro.models.ModelCatalog
import dev.soupslurpr.transcribro.models.ModelEntry
import dev.soupslurpr.transcribro.models.ModelKind
import dev.soupslurpr.transcribro.ui.reusablecomposables.ScreenLazyColumn

@Composable
fun ModelsScreen() {
    val app = LocalContext.current.applicationContext as android.app.Application
    val viewModel: ModelsViewModel = viewModel(factory = ModelsViewModel.Factory(app))

    val statesById by viewModel.statesById.collectAsState()
    val installedState by viewModel.installedState.collectAsState()

    // Visible rows are everything except "not installed and idle" — i.e. installed,
    // currently downloading, or in a failed state. Catalog entries the user has never
    // touched live in the per-section install dialogs.
    val visibleByKind = ModelCatalog.entries
        .filter { entry ->
            val state = statesById[entry.id]
            state !is ModelUiState.Idle || state.installed
        }
        .groupBy { it.kind }

    // null = no picker open; non-null = picker filtered to this kind.
    var pickerKind: ModelKind? by remember { mutableStateOf(null) }

    ScreenLazyColumn(modifier = Modifier.fillMaxSize()) {
        item {
            DiskUsageSummary(totalBytes = installedState.totalSizeBytes)
        }

        for (kind in ModelKind.values()) {
            item(key = "header_${kind.name}") {
                SettingsCategory(kind.displayName())
            }
            val entries = visibleByKind[kind] ?: emptyList()
            for (entry in entries) {
                item(key = "entry_${entry.id}") {
                    val state = statesById[entry.id] ?: ModelUiState.Idle(false)
                    ModelRow(
                        entry = entry,
                        state = state,
                        onCancel = { viewModel.cancel(entry) },
                        onUninstall = { viewModel.uninstall(entry) },
                        onRetry = { viewModel.install(entry) },
                        onDismissFailure = { viewModel.dismissFailure(entry) },
                    )
                }
            }
            // Pickable count for this kind — used to decide whether the install action
            // should be enabled or replaced with an "all installed" hint.
            val pickableForKind = ModelCatalog.forKind(kind).count { entry ->
                val s = statesById[entry.id]
                s is ModelUiState.Idle && !s.installed
            }
            item(key = "install_${kind.name}") {
                InstallNewRow(
                    kind = kind,
                    enabled = pickableForKind > 0,
                    onClick = { pickerKind = kind },
                )
            }
        }
    }

    pickerKind?.let { kind ->
        InstallPickerDialog(
            kind = kind,
            statesById = statesById,
            onDismiss = { pickerKind = null },
            onInstall = { entry ->
                viewModel.install(entry)
                pickerKind = null
            },
        )
    }
}

@Composable
private fun DiskUsageSummary(totalBytes: Long) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp),
    ) {
        Text(
            text = "Disk usage: ${formatBytes(totalBytes)}",
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            text = "Installed models live in this app's private storage and are deleted when the app is uninstalled.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun InstallNewRow(kind: ModelKind, enabled: Boolean, onClick: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .let { if (enabled) it.clickable(onClick = onClick) else it }
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = androidx.compose.ui.Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        val tint = if (enabled) MaterialTheme.colorScheme.primary
        else MaterialTheme.colorScheme.onSurfaceVariant
        androidx.compose.material3.Icon(
            imageVector = Icons.Filled.Add,
            contentDescription = null,
            tint = tint,
        )
        Column {
            Text(
                if (enabled) "Install new ${kind.shortName()}"
                else "All ${kind.shortName()} models installed",
                fontWeight = FontWeight.SemiBold,
                color = tint,
            )
            if (enabled) {
                Text(
                    "Pick a ${kind.shortName()} model from the catalog",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun ModelRow(
    entry: ModelEntry,
    state: ModelUiState,
    onCancel: () -> Unit,
    onUninstall: () -> Unit,
    onRetry: () -> Unit,
    onDismissFailure: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp),
    ) {
        Text(entry.displayName, fontWeight = FontWeight.SemiBold)
        if (entry.description != null) {
            Text(
                entry.description,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Text(
            "Size: ${formatBytes(entry.sizeBytes)}",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Spacer(Modifier.size(8.dp))
        when (state) {
            is ModelUiState.Idle -> {
                // visibleEntries already filters out NotInstalled+Idle, so we only get
                // here for installed entries.
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        "Installed",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.primary,
                    )
                    Spacer(Modifier.size(8.dp))
                    OutlinedButton(onClick = onUninstall) { Text("Uninstall") }
                }
            }

            is ModelUiState.Downloading -> {
                val frac = state.fraction
                if (frac != null) {
                    LinearProgressIndicator(
                        progress = { frac },
                        modifier = Modifier.fillMaxWidth(),
                    )
                } else {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }
                Spacer(Modifier.size(4.dp))
                val label = when (state.phase) {
                    ModelUiState.Downloading.Phase.Connecting -> "Connecting…"
                    ModelUiState.Downloading.Phase.Downloading ->
                        if (state.totalBytes != null) {
                            "${formatBytes(state.bytesDownloaded)} / ${formatBytes(state.totalBytes)}"
                        } else {
                            "${formatBytes(state.bytesDownloaded)} downloaded"
                        }
                    ModelUiState.Downloading.Phase.Verifying -> "Verifying…"
                }
                Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
                    Text(label, style = MaterialTheme.typography.labelSmall)
                    Spacer(Modifier.size(12.dp))
                    TextButton(onClick = onCancel) { Text("Cancel") }
                }
            }

            is ModelUiState.Failed -> {
                Text(
                    state.message,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
                Spacer(Modifier.size(4.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = onRetry) { Text("Retry") }
                    TextButton(onClick = onDismissFailure) { Text("Dismiss") }
                }
            }
        }
    }
}

@Composable
private fun InstallPickerDialog(
    kind: ModelKind,
    statesById: Map<String, ModelUiState>,
    onDismiss: () -> Unit,
    onInstall: (ModelEntry) -> Unit,
) {
    // Pickable = of this kind AND not installed AND not currently downloading or failed.
    // Failed/downloading entries already have an affordance in the main list, so we
    // exclude them here to avoid two install paths pointing at the same row.
    val pickable = ModelCatalog.forKind(kind).filter { entry ->
        val s = statesById[entry.id]
        s is ModelUiState.Idle && !s.installed
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(onClick = onDismiss) { Text("Close") }
        },
        title = { Text("Install ${kind.shortName()}") },
        text = {
            if (pickable.isEmpty()) {
                Text("All ${kind.shortName()} models are already installed.")
            } else {
                // The dialog content area gets a bounded height so the LazyColumn can
                // scroll without expanding the dialog past the screen.
                LazyColumn(
                    modifier = Modifier.heightIn(max = 480.dp),
                ) {
                    for (entry in pickable) {
                        item(key = "picker_entry_${entry.id}") {
                            PickerRow(entry = entry, onClick = { onInstall(entry) })
                        }
                    }
                }
            }
        },
    )
}

@Composable
private fun PickerRow(entry: ModelEntry, onClick: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(vertical = 8.dp),
    ) {
        Text(entry.displayName, fontWeight = FontWeight.SemiBold)
        if (entry.description != null) {
            Text(
                entry.description,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Text(
            "Size: ${formatBytes(entry.sizeBytes)}",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

private fun ModelKind.displayName(): String = when (this) {
    ModelKind.WhisperTranscriber -> "Transcription (Whisper)"
    ModelKind.VoiceActivityDetector -> "Voice Activity Detection"
    ModelKind.SpeakerSegmenter -> "Speaker Segmentation"
    ModelKind.SpeakerEmbedder -> "Speaker Embedding"
}

/** Lowercase, sentence-friendly noun for use in inline copy ("Install new <noun>"). */
private fun ModelKind.shortName(): String = when (this) {
    ModelKind.WhisperTranscriber -> "transcription model"
    ModelKind.VoiceActivityDetector -> "voice activity model"
    ModelKind.SpeakerSegmenter -> "speaker segmenter"
    ModelKind.SpeakerEmbedder -> "speaker embedder"
}

private fun formatBytes(bytes: Long): String {
    if (bytes <= 0) return "0 B"
    val units = arrayOf("B", "KB", "MB", "GB")
    var value = bytes.toDouble()
    var idx = 0
    while (value >= 1024.0 && idx < units.lastIndex) {
        value /= 1024.0
        idx++
    }
    return if (idx == 0) "${value.toLong()} ${units[idx]}" else "%.1f %s".format(value, units[idx])
}
