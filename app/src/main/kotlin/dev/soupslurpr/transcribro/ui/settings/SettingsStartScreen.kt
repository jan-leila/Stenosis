package dev.soupslurpr.transcribro.ui.settings

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyListScope
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ExitToApp
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.soupslurpr.transcribro.R
import dev.soupslurpr.transcribro.dataStore
import dev.soupslurpr.transcribro.models.ModelKind
import dev.soupslurpr.transcribro.preferences.PreferencesUiState
import dev.soupslurpr.transcribro.preferences.PreferencesViewModel
import dev.soupslurpr.transcribro.ui.reusablecomposables.ScreenLazyColumn

@Composable
fun SettingsStartScreen(
    onClickModels: () -> Unit,
    onClickLicense: () -> Unit,
    onClickPrivacyPolicy: () -> Unit,
    onClickCredits: () -> Unit,
) {
    val preferencesViewModel: PreferencesViewModel = viewModel(
        factory = PreferencesViewModel.PreferencesViewModelFactory(LocalContext.current.dataStore)
    )

    val preferencesUiState by preferencesViewModel.uiState.collectAsState()

    val localUriHandler = LocalUriHandler.current

    // Independent expand state per Advanced section so a power-user can tune one
    // mode without scrolling past the other's tunables.
    var conversationAdvancedExpanded by rememberSaveable { mutableStateOf(false) }
    var keyboardAdvancedExpanded by rememberSaveable { mutableStateOf(false) }

    ScreenLazyColumn(
        modifier = Modifier
            .fillMaxSize()
    ) {
        item {
            SettingsIconItem(
                name = stringResource(id = R.string.models_setting_name),
                description = stringResource(id = R.string.models_setting_description),
                icon = Icons.AutoMirrored.Filled.KeyboardArrowRight,
                onClick = onClickModels,
            )
        }
        item {
            SettingsCategory(
                stringResource(R.string.theme)
            )
        }
        item {
            val preference = preferencesUiState.pitchBlackBackground
            SettingsSwitchItem(
                name = stringResource(id = R.string.pitch_black_background_setting_name),
                description = stringResource(id = R.string.pitch_black_background_setting_description),
                checked = preference.second.value,
                onCheckedChange = {
                    preferencesViewModel.setPreference(preference.first, it)
                }
            )
        }

        item {
            SettingsCategory(
                stringResource(R.string.conversation_settings_category)
            )
        }
        item {
            val preference = preferencesUiState.diarizationEnabled
            SettingsSwitchItem(
                name = stringResource(id = R.string.diarization_setting_name),
                description = stringResource(id = R.string.diarization_setting_description),
                checked = preference.second.value,
                onCheckedChange = {
                    preferencesViewModel.setPreference(preference.first, it)
                }
            )
        }
        advancedSection(
            expanded = conversationAdvancedExpanded,
            onToggle = { conversationAdvancedExpanded = !conversationAdvancedExpanded },
        ) {
            item {
                val preference = preferencesUiState.conversationStreamingTranscription
                SettingsSwitchItem(
                    name = stringResource(id = R.string.streaming_transcription_setting_name),
                    description = stringResource(id = R.string.streaming_transcription_setting_description),
                    checked = preference.second.value,
                    onCheckedChange = {
                        preferencesViewModel.setPreference(preference.first, it)
                    },
                )
            }
            item {
                val preference = preferencesUiState.conversationNoiseSuppressionEnabled
                SettingsSwitchItem(
                    name = stringResource(id = R.string.noise_suppression_setting_name),
                    description = stringResource(id = R.string.noise_suppression_setting_description),
                    checked = preference.second.value,
                    onCheckedChange = {
                        preferencesViewModel.setPreference(preference.first, it)
                    },
                )
            }
            item {
                val preference = preferencesUiState.conversationMinSilenceMs
                SettingsIntItem(
                    name = stringResource(id = R.string.min_silence_setting_name),
                    description = stringResource(id = R.string.min_silence_setting_description),
                    value = preference.second.value,
                    defaultValue = PreferencesUiState.DEFAULT_MIN_SILENCE_MS,
                    onValueChange = { preferencesViewModel.setPreference(preference.first, it) },
                    onReset = { preferencesViewModel.resetPreference(preference.first) },
                )
            }
            item {
                val preference = preferencesUiState.conversationVadStartThreshold
                SettingsFloatItem(
                    name = stringResource(id = R.string.vad_start_threshold_setting_name),
                    description = stringResource(id = R.string.vad_start_threshold_setting_description),
                    value = preference.second.value,
                    defaultValue = PreferencesUiState.DEFAULT_VAD_START_THRESHOLD,
                    onValueChange = { preferencesViewModel.setPreference(preference.first, it) },
                    onReset = { preferencesViewModel.resetPreference(preference.first) },
                )
            }
            item {
                val preference = preferencesUiState.conversationVadEndThreshold
                SettingsFloatItem(
                    name = stringResource(id = R.string.vad_end_threshold_setting_name),
                    description = stringResource(id = R.string.vad_end_threshold_setting_description),
                    value = preference.second.value,
                    defaultValue = PreferencesUiState.DEFAULT_VAD_END_THRESHOLD,
                    onValueChange = { preferencesViewModel.setPreference(preference.first, it) },
                    onReset = { preferencesViewModel.resetPreference(preference.first) },
                )
            }
            item {
                val preference = preferencesUiState.conversationCosineThreshold
                SettingsFloatItem(
                    name = stringResource(id = R.string.cosine_threshold_setting_name),
                    description = stringResource(id = R.string.cosine_threshold_setting_description),
                    value = preference.second.value,
                    defaultValue = PreferencesUiState.DEFAULT_COSINE_THRESHOLD,
                    onValueChange = { preferencesViewModel.setPreference(preference.first, it) },
                    onReset = { preferencesViewModel.resetPreference(preference.first) },
                )
            }
            item {
                val preference = preferencesUiState.conversationVadModelId
                ModelPicker(
                    kind = ModelKind.VoiceActivityDetector,
                    name = stringResource(id = R.string.vad_model_picker_name),
                    description = stringResource(id = R.string.vad_model_picker_description),
                    selectedId = preference.second.value,
                    onSelect = { preferencesViewModel.setPreference(preference.first, it) },
                    allowNone = false,
                )
            }
            item {
                val preference = preferencesUiState.conversationWhisperModelId
                ModelPicker(
                    kind = ModelKind.WhisperTranscriber,
                    name = stringResource(id = R.string.whisper_model_picker_name),
                    description = stringResource(id = R.string.whisper_model_picker_description),
                    selectedId = preference.second.value,
                    onSelect = { preferencesViewModel.setPreference(preference.first, it) },
                )
            }
            item {
                val preference = preferencesUiState.conversationWhisperStreamingModelId
                ModelPicker(
                    kind = ModelKind.WhisperTranscriber,
                    name = stringResource(id = R.string.whisper_streaming_model_picker_name),
                    description = stringResource(id = R.string.whisper_streaming_model_picker_description),
                    selectedId = preference.second.value,
                    onSelect = { preferencesViewModel.setPreference(preference.first, it) },
                )
            }
            item {
                val preference = preferencesUiState.conversationSegmenterModelId
                ModelPicker(
                    kind = ModelKind.SpeakerSegmenter,
                    name = stringResource(id = R.string.segmenter_model_picker_name),
                    description = stringResource(id = R.string.segmenter_model_picker_description),
                    selectedId = preference.second.value,
                    onSelect = { preferencesViewModel.setPreference(preference.first, it) },
                )
            }
            item {
                val preference = preferencesUiState.conversationEmbedderModelId
                ModelPicker(
                    kind = ModelKind.SpeakerEmbedder,
                    name = stringResource(id = R.string.embedder_model_picker_name),
                    description = stringResource(id = R.string.embedder_model_picker_description),
                    selectedId = preference.second.value,
                    onSelect = { preferencesViewModel.setPreference(preference.first, it) },
                )
            }
        }

        item {
            SettingsCategory(
                stringResource(R.string.keyboard_settings_category)
            )
        }
        item {
            val preference = preferencesUiState.autoSwitchToPreviousInputMethod
            SettingsSwitchItem(
                name = stringResource(id = R.string.auto_switch_to_previous_input_method_setting_name),
                description = stringResource(id = R.string.auto_switch_to_previous_input_method_setting_description),
                checked = preference.second.value,
                onCheckedChange = {
                    preferencesViewModel.setPreference(preference.first, it)
                }
            )
        }
        item {
            val preference = preferencesUiState.autoStopRecognition
            SettingsSwitchItem(
                name = stringResource(id = R.string.auto_stop_recognition_setting_name),
                description = stringResource(id = R.string.auto_stop_recognition_setting_description),
                checked = preference.second.value,
                onCheckedChange = {
                    preferencesViewModel.setPreference(preference.first, it)
                }
            )
        }
        item {
            val preference = preferencesUiState.autoStartRecognition
            SettingsSwitchItem(
                name = stringResource(id = R.string.auto_start_recognition_setting_name),
                description = stringResource(id = R.string.auto_start_recognition_setting_description),
                checked = preference.second.value,
                onCheckedChange = {
                    preferencesViewModel.setPreference(preference.first, it)
                }
            )
        }
        item {
            val preference = preferencesUiState.autoSendTranscription
            SettingsSwitchItem(
                name = stringResource(id = R.string.auto_send_transcription_setting_name),
                description = stringResource(id = R.string.auto_send_transcription_setting_description),
                checked = preference.second.value,
                onCheckedChange = {
                    preferencesViewModel.setPreference(preference.first, it)
                }
            )
        }
        advancedSection(
            expanded = keyboardAdvancedExpanded,
            onToggle = { keyboardAdvancedExpanded = !keyboardAdvancedExpanded },
        ) {
            item {
                val preference = preferencesUiState.keyboardStreamingTranscription
                SettingsSwitchItem(
                    name = stringResource(id = R.string.streaming_transcription_setting_name),
                    description = stringResource(id = R.string.streaming_transcription_setting_description),
                    checked = preference.second.value,
                    onCheckedChange = {
                        preferencesViewModel.setPreference(preference.first, it)
                    },
                )
            }
            item {
                val preference = preferencesUiState.keyboardNoiseSuppressionEnabled
                SettingsSwitchItem(
                    name = stringResource(id = R.string.noise_suppression_setting_name),
                    description = stringResource(id = R.string.noise_suppression_setting_description),
                    checked = preference.second.value,
                    onCheckedChange = {
                        preferencesViewModel.setPreference(preference.first, it)
                    },
                )
            }
            item {
                val preference = preferencesUiState.keyboardMinSilenceMs
                SettingsIntItem(
                    name = stringResource(id = R.string.min_silence_setting_name),
                    description = stringResource(id = R.string.min_silence_setting_description),
                    value = preference.second.value,
                    defaultValue = PreferencesUiState.DEFAULT_MIN_SILENCE_MS,
                    onValueChange = { preferencesViewModel.setPreference(preference.first, it) },
                    onReset = { preferencesViewModel.resetPreference(preference.first) },
                )
            }
            item {
                val preference = preferencesUiState.keyboardVadStartThreshold
                SettingsFloatItem(
                    name = stringResource(id = R.string.vad_start_threshold_setting_name),
                    description = stringResource(id = R.string.vad_start_threshold_setting_description),
                    value = preference.second.value,
                    defaultValue = PreferencesUiState.DEFAULT_VAD_START_THRESHOLD,
                    onValueChange = { preferencesViewModel.setPreference(preference.first, it) },
                    onReset = { preferencesViewModel.resetPreference(preference.first) },
                )
            }
            item {
                val preference = preferencesUiState.keyboardVadEndThreshold
                SettingsFloatItem(
                    name = stringResource(id = R.string.vad_end_threshold_setting_name),
                    description = stringResource(id = R.string.vad_end_threshold_setting_description),
                    value = preference.second.value,
                    defaultValue = PreferencesUiState.DEFAULT_VAD_END_THRESHOLD,
                    onValueChange = { preferencesViewModel.setPreference(preference.first, it) },
                    onReset = { preferencesViewModel.resetPreference(preference.first) },
                )
            }
            item {
                val preference = preferencesUiState.keyboardVadModelId
                ModelPicker(
                    kind = ModelKind.VoiceActivityDetector,
                    name = stringResource(id = R.string.vad_model_picker_name),
                    description = stringResource(id = R.string.vad_model_picker_description),
                    selectedId = preference.second.value,
                    onSelect = { preferencesViewModel.setPreference(preference.first, it) },
                    allowNone = false,
                )
            }
            item {
                val preference = preferencesUiState.keyboardWhisperModelId
                ModelPicker(
                    kind = ModelKind.WhisperTranscriber,
                    name = stringResource(id = R.string.whisper_model_picker_name),
                    description = stringResource(id = R.string.whisper_model_picker_description),
                    selectedId = preference.second.value,
                    onSelect = { preferencesViewModel.setPreference(preference.first, it) },
                )
            }
            item {
                val preference = preferencesUiState.keyboardWhisperStreamingModelId
                ModelPicker(
                    kind = ModelKind.WhisperTranscriber,
                    name = stringResource(id = R.string.whisper_streaming_model_picker_name),
                    description = stringResource(id = R.string.whisper_streaming_model_picker_description),
                    selectedId = preference.second.value,
                    onSelect = { preferencesViewModel.setPreference(preference.first, it) },
                )
            }
        }

        item {
            SettingsCategory(
                stringResource(R.string.about_setting_category)
            )
        }
        item {
            SettingsIconItem(
                name = stringResource(id = R.string.view_source_code_setting_name),
                description = stringResource(id = R.string.view_source_code_setting_description),
                icon = Icons.AutoMirrored.Filled.ExitToApp,
                onClick = {
                    localUriHandler.openUri("https://git.jan-leila.com/jan-leila/stenographerSis")
                }
            )
        }
        item {
            SettingsIconItem(
                name = stringResource(id = R.string.license_setting_name),
                description = stringResource(id = R.string.license_setting_description),
                icon = Icons.Filled.Info,
                onClick = onClickLicense
            )
        }
        item {
            SettingsIconItem(
                name = stringResource(id = R.string.privacy_policy_setting_name),
                description = stringResource(id = R.string.privacy_policy_setting_description),
                icon = Icons.Filled.Info,
                onClick = onClickPrivacyPolicy
            )
        }
        item {
            SettingsIconItem(
                name = stringResource(id = R.string.credits_setting_name),
                description = stringResource(id = R.string.credits_setting_description),
                icon = Icons.Filled.Info,
                onClick = onClickCredits
            )
        }
    }
}

/**
 * Emit a collapsible "Advanced" header followed (when expanded) by the caller-supplied
 * content items. Implemented as a [LazyListScope] extension so the inner items become
 * direct children of the surrounding lazy column — they participate in scrolling and
 * recycling on the same terms as the top-level items, rather than being squeezed into
 * a single nested-scrolling row.
 */
private fun LazyListScope.advancedSection(
    expanded: Boolean,
    onToggle: () -> Unit,
    content: LazyListScope.() -> Unit,
) {
    item {
        AdvancedSectionHeader(expanded = expanded, onToggle = onToggle)
    }
    if (expanded) content()
}

@Composable
private fun AdvancedSectionHeader(expanded: Boolean, onToggle: () -> Unit) {
    val label = stringResource(
        if (expanded) R.string.advanced_setting_hide else R.string.advanced_setting_show,
    )
    ListItem(
        modifier = Modifier.clickable(onClick = onToggle),
        headlineContent = {
            Text(
                stringResource(R.string.advanced_setting_category),
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.primary,
            )
        },
        trailingContent = {
            Icon(
                imageVector = if (expanded) Icons.Filled.KeyboardArrowUp else Icons.Filled.KeyboardArrowDown,
                contentDescription = label,
            )
        },
    )
}

@Composable
fun SettingsCategory(category: String) {
    Text(
        text = category,
        modifier = Modifier.padding(top = 8.dp),
        textAlign = TextAlign.Center,
        color = MaterialTheme.colorScheme.primary,
        style = MaterialTheme.typography.bodyMedium,
        fontWeight = FontWeight.Bold
    )
}

@Composable
fun SettingsSwitchItem(
    modifier: Modifier = Modifier,
    name: String,
    description: String? = null,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    ListItem(
        modifier = modifier
            .toggleable(
                value = checked,
                onValueChange = { onCheckedChange(it) }
            ),
        headlineContent = {
            Text(
                name,
                fontWeight = FontWeight.SemiBold
            )
        },
        supportingContent = run {
            if (description != null) {
                { Text(description) }
            } else {
                null
            }
        },
        trailingContent = {
            Switch(
                checked = checked,
                onCheckedChange = null
            )
        }
    )
}

@Composable
fun SettingsIconItem(
    modifier: Modifier = Modifier,
    name: String,
    description: String? = null,
    icon: ImageVector,
    onClick: () -> Unit
) {
    ListItem(
        modifier = modifier
            .clickable(
                onClick = onClick
            ),
        headlineContent = {
            Text(
                name,
                fontWeight = FontWeight.SemiBold
            )
        },
        supportingContent = run {
            if (description != null) {
                { Text(description) }
            } else {
                null
            }
        },
        trailingContent = {
            Icon(
                imageVector = icon,
                contentDescription = null
            )
        }
    )
}

/**
 * Integer-valued setting row. The edit field is bound to a local string state so the
 * user can briefly type an invalid prefix (e.g. an empty box mid-edit) without the
 * stored value flapping. The stored preference only updates when the input parses to
 * an [Int]; the [onReset] callback removes the key entirely so the next read falls
 * back to the compile-time default.
 */
@Composable
fun SettingsIntItem(
    name: String,
    description: String? = null,
    value: Int,
    defaultValue: Int,
    onValueChange: (Int) -> Unit,
    onReset: () -> Unit,
) {
    NumberSettingRow(
        name = name,
        description = description,
        valueText = value.toString(),
        defaultText = defaultValue.toString(),
        keyboardType = KeyboardType.Number,
        parseAndApply = { text -> text.toIntOrNull()?.let(onValueChange) },
        onReset = onReset,
    )
}

/**
 * Floating-point setting row. See [SettingsIntItem]; the only difference is the
 * keyboard hint and the parsing function.
 */
@Composable
fun SettingsFloatItem(
    name: String,
    description: String? = null,
    value: Float,
    defaultValue: Float,
    onValueChange: (Float) -> Unit,
    onReset: () -> Unit,
) {
    NumberSettingRow(
        name = name,
        description = description,
        valueText = value.toString(),
        defaultText = defaultValue.toString(),
        keyboardType = KeyboardType.Decimal,
        parseAndApply = { text -> text.toFloatOrNull()?.let(onValueChange) },
        onReset = onReset,
    )
}

@Composable
private fun NumberSettingRow(
    name: String,
    description: String?,
    valueText: String,
    defaultText: String,
    keyboardType: KeyboardType,
    parseAndApply: (String) -> Unit,
    onReset: () -> Unit,
) {
    // The edit buffer is keyed by the persisted value so external resets (or another
    // tab updating the same key) propagate into the field; user edits in-between
    // persist immediately on a successful parse, so this stays in sync without
    // fighting the user's typing.
    var draft by remember(valueText) { mutableStateOf(valueText) }
    ListItem(
        headlineContent = {
            Text(name, fontWeight = FontWeight.SemiBold)
        },
        supportingContent = {
            Column {
                if (description != null) {
                    Text(description)
                }
                OutlinedTextField(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 8.dp),
                    value = draft,
                    onValueChange = { input ->
                        draft = input
                        parseAndApply(input)
                    },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = keyboardType),
                    supportingText = { Text("default $defaultText") },
                )
            }
        },
        trailingContent = {
            IconButton(onClick = onReset) {
                Icon(
                    imageVector = Icons.Filled.Refresh,
                    contentDescription = stringResource(
                        R.string.reset_to_default_content_description,
                    ),
                )
            }
        },
    )
}

