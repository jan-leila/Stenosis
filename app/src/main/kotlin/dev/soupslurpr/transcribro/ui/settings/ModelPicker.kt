package dev.soupslurpr.transcribro.ui.settings

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.soupslurpr.transcribro.models.InstalledModelsRepository
import dev.soupslurpr.transcribro.models.ModelCatalog
import dev.soupslurpr.transcribro.models.ModelKind

/**
 * Picker for selecting which installed model of a given [kind] should be used by some
 * feature. Tapping the row opens a dialog with all installed entries of [kind] (plus a
 * "None" option). Catalog entries that are not yet installed appear disabled with an
 * "(install in Models)" suffix — they remind the user where to go but aren't selectable.
 *
 * [name] / [description] are the row's headline/subtitle; the trailing value shows the
 * currently selected entry's display name (or "(not selected)").
 */
@Composable
fun ModelPicker(
    kind: ModelKind,
    name: String,
    description: String?,
    selectedId: String?,
    onSelect: (String?) -> Unit,
    allowNone: Boolean = true,
) {
    val context = LocalContext.current
    val installedRepo = remember { InstalledModelsRepository(context) }
    val installed by installedRepo.state.collectAsState()

    var dialogOpen by remember { mutableStateOf(false) }

    val entriesForKind = ModelCatalog.forKind(kind)
    val selectedEntry = selectedId?.let { ModelCatalog.findById(it) }
    val selectedDisplay = when {
        selectedEntry == null -> "(not selected)"
        !installed.byId.containsKey(selectedEntry.id) -> "${selectedEntry.displayName} (not installed)"
        else -> selectedEntry.displayName
    }

    ListItem(
        modifier = Modifier.clickable { dialogOpen = true },
        headlineContent = { Text(name, fontWeight = FontWeight.SemiBold) },
        supportingContent = {
            Column {
                if (description != null) Text(description)
                Text(
                    selectedDisplay,
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.primary,
                )
            }
        },
    )

    if (dialogOpen) {
        AlertDialog(
            onDismissRequest = { dialogOpen = false },
            confirmButton = {
                TextButton(onClick = { dialogOpen = false }) { Text("Done") }
            },
            title = { Text(name) },
            text = {
                Column {
                    if (allowNone) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable {
                                    onSelect(null)
                                    dialogOpen = false
                                }
                                .padding(vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            RadioButton(
                                selected = selectedId == null,
                                onClick = {
                                    onSelect(null)
                                    dialogOpen = false
                                },
                            )
                            Text("None")
                        }
                    }
                    for (entry in entriesForKind) {
                        val isInstalled = installed.byId.containsKey(entry.id)
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .let { m ->
                                    if (isInstalled) m.clickable {
                                        onSelect(entry.id)
                                        dialogOpen = false
                                    } else m
                                }
                                .padding(vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            RadioButton(
                                selected = selectedId == entry.id,
                                enabled = isInstalled,
                                onClick = if (isInstalled) {
                                    {
                                        onSelect(entry.id)
                                        dialogOpen = false
                                    }
                                } else null,
                            )
                            Column {
                                Text(
                                    entry.displayName,
                                    color = if (isInstalled) {
                                        MaterialTheme.colorScheme.onSurface
                                    } else {
                                        MaterialTheme.colorScheme.onSurfaceVariant
                                    },
                                )
                                if (!isInstalled) {
                                    Text(
                                        "(install in Models)",
                                        style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    )
                                }
                            }
                        }
                    }
                }
            },
        )
    }
}
