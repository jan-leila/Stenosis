package dev.soupslurpr.transcribro.ui.settings

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyListScope
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ExitToApp
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.NetworkCheck
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.VisibilityOff
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.soupslurpr.transcribro.R
import dev.soupslurpr.transcribro.dataStore
import dev.soupslurpr.transcribro.models.ModelKind
import dev.soupslurpr.transcribro.preferences.PreferencesUiState
import dev.soupslurpr.transcribro.preferences.PreferencesViewModel
import dev.soupslurpr.transcribro.recognitionservice.remote.RemoteServerConfig
import dev.soupslurpr.transcribro.recognitionservice.remote.RemoteTranscriptionEngine
import dev.soupslurpr.transcribro.recognitionservice.remote.ServerUrlValidation
import dev.soupslurpr.transcribro.recognitionservice.remote.validateServerUrl
import dev.soupslurpr.transcribro.ui.reusablecomposables.ScreenLazyColumn
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch

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
        item {
            val preference = preferencesUiState.conversationServerTranscription
            SettingsSwitchItem(
                name = stringResource(id = R.string.server_transcription_setting_name),
                description = stringResource(id = R.string.server_transcription_setting_description),
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
        item {
            val preference = preferencesUiState.keyboardServerTranscription
            SettingsSwitchItem(
                name = stringResource(id = R.string.server_transcription_setting_name),
                description = stringResource(id = R.string.server_transcription_setting_description),
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

        // Shared by both modes: there is one server, and each mode's switch above
        // decides whether that mode uses it.
        item {
            SettingsCategory(
                stringResource(R.string.server_settings_category)
            )
        }
        item {
            val preference = preferencesUiState.serverUrl
            val validation = validateServerUrl(preference.second.value)
            SettingsTextItem(
                name = stringResource(id = R.string.server_url_setting_name),
                description = stringResource(id = R.string.server_url_setting_description),
                value = preference.second.value,
                onValueChange = { preferencesViewModel.setPreference(preference.first, it.trim()) },
                placeholder = "https://example.com/v1/audio/transcriptions",
                keyboardType = KeyboardType.Uri,
                // Don't nag about an empty field: that's the normal "not using a server"
                // state. Only flag a URL the user actually typed that we'd refuse to use.
                errorText = (validation as? ServerUrlValidation.Invalid)?.reason
                    ?.takeIf { preference.second.value.isNotBlank() },
            )
        }
        item {
            val preference = preferencesUiState.serverModel
            SettingsTextItem(
                name = stringResource(id = R.string.server_model_setting_name),
                description = stringResource(id = R.string.server_model_setting_description),
                value = preference.second.value,
                onValueChange = { preferencesViewModel.setPreference(preference.first, it.trim()) },
            )
        }
        item {
            val preference = preferencesUiState.serverToken
            SettingsTextItem(
                name = stringResource(id = R.string.server_token_setting_name),
                description = stringResource(id = R.string.server_token_setting_description),
                value = preference.second.value,
                onValueChange = { preferencesViewModel.setPreference(preference.first, it.trim()) },
                keyboardType = KeyboardType.Password,
                secret = true,
            )
        }
        item {
            ServerTestConnectionItem(
                url = preferencesUiState.serverUrl.second.value,
                model = preferencesUiState.serverModel.second.value,
                token = preferencesUiState.serverToken.second.value,
            )
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

/**
 * Free-text setting row, styled like [NumberSettingRow] minus the reset button (an
 * empty field *is* the default for every text setting so far).
 *
 * The draft is decoupled from the persisted value while the field has focus. Every
 * keystroke is persisted, and DataStore echoes each write back a moment later; if the
 * draft re-synced on every echo (as [NumberSettingRow] does) a fast typist would see
 * characters vanish when a stale echo overwrote a newer draft. So we only adopt the
 * persisted value when the field isn't focused — which covers the initial async load
 * from DataStore, the case that actually matters.
 *
 * [secret] masks the input (for tokens) with a toggle to reveal it, so the user can
 * check a pasted value without it sitting on screen by default.
 */
@Composable
fun SettingsTextItem(
    name: String,
    description: String? = null,
    value: String,
    onValueChange: (String) -> Unit,
    placeholder: String? = null,
    errorText: String? = null,
    keyboardType: KeyboardType = KeyboardType.Text,
    secret: Boolean = false,
) {
    var draft by remember { mutableStateOf(value) }
    var focused by remember { mutableStateOf(false) }
    var revealed by rememberSaveable { mutableStateOf(false) }
    LaunchedEffect(value, focused) {
        if (!focused) draft = value
    }
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
                        .padding(top = 8.dp)
                        .onFocusChanged { focused = it.isFocused },
                    value = draft,
                    onValueChange = { input ->
                        draft = input
                        onValueChange(input)
                    },
                    singleLine = true,
                    placeholder = placeholder?.let { { Text(it) } },
                    isError = errorText != null,
                    supportingText = errorText?.let { { Text(it) } },
                    keyboardOptions = KeyboardOptions(
                        keyboardType = keyboardType,
                        autoCorrectEnabled = false,
                    ),
                    visualTransformation = if (secret && !revealed) {
                        PasswordVisualTransformation()
                    } else {
                        VisualTransformation.None
                    },
                    trailingIcon = if (secret) {
                        {
                            IconButton(onClick = { revealed = !revealed }) {
                                Icon(
                                    imageVector = if (revealed) {
                                        Icons.Filled.VisibilityOff
                                    } else {
                                        Icons.Filled.Visibility
                                    },
                                    contentDescription = stringResource(
                                        if (revealed) R.string.hide_secret_content_description
                                        else R.string.show_secret_content_description,
                                    ),
                                )
                            }
                        }
                    } else null,
                )
            }
        },
    )
}

/**
 * "Test connection" row: sends one second of silence through exactly the same request
 * path a real session uses (same headers, multipart body and response parsing), so a
 * green result here means recording will work, not merely that the host answers.
 * Silence usually transcribes to an empty string, which still counts as success.
 */
@Composable
private fun ServerTestConnectionItem(url: String, model: String, token: String) {
    val scope = rememberCoroutineScope()
    var running by remember { mutableStateOf(false) }
    var result by remember { mutableStateOf<String?>(null) }
    var resultIsError by remember { mutableStateOf(false) }
    val successText = stringResource(R.string.server_test_connection_success)

    ListItem(
        modifier = Modifier.clickable(enabled = !running) {
            val validation = validateServerUrl(url)
            if (validation is ServerUrlValidation.Invalid) {
                result = validation.reason
                resultIsError = true
                return@clickable
            }
            running = true
            result = null
            scope.launch {
                val engine = RemoteTranscriptionEngine(
                    RemoteServerConfig(
                        url = (validation as ServerUrlValidation.Valid).url,
                        model = model,
                        bearerToken = token,
                    ),
                )
                // The engine moves the request onto Dispatchers.IO itself; this
                // coroutine only touches Compose state, on the main thread.
                try {
                    engine.transcribeAudio(RemoteTranscriptionEngine.TEST_CONNECTION_AUDIO)
                    result = successText
                    resultIsError = false
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    result = e.message ?: e::class.java.simpleName
                    resultIsError = true
                } finally {
                    running = false
                }
            }
        },
        headlineContent = {
            Text(
                stringResource(R.string.server_test_connection_name),
                fontWeight = FontWeight.SemiBold,
            )
        },
        supportingContent = {
            val text = result
            when {
                running -> Text(stringResource(R.string.server_test_connection_running))
                text != null -> Text(
                    text,
                    color = if (resultIsError) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary,
                )
                else -> Text(stringResource(R.string.server_test_connection_description))
            }
        },
        trailingContent = {
            if (running) {
                CircularProgressIndicator(modifier = Modifier.size(24.dp), strokeWidth = 2.dp)
            } else {
                Icon(imageVector = Icons.Filled.NetworkCheck, contentDescription = null)
            }
        },
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

