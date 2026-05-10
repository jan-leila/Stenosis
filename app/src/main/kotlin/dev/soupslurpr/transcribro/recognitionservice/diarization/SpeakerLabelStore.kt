package dev.soupslurpr.transcribro.recognitionservice.diarization

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Holds user-facing display names for the integer speaker ids produced by
 * [SpeakerRegistry]. Decoupled from the registry so:
 *   - The clustering layer is pure math (ints in, ints out — no UI strings to translate).
 *   - Renames (manual rename gesture, or a post-session LLM pass that maps speaker ids
 *     to actual names) are a single state update with no need to rewrite historical
 *     transcript bundles.
 *   - The UI subscribes to [labels] and re-renders when names change.
 *
 * Default name for an unknown id is `"Speaker ${id + 1}"`.
 */
class SpeakerLabelStore {
    private val _labels = MutableStateFlow<Map<Int, String>>(emptyMap())

    /** Current id-to-name mapping. UI renders by looking up speaker ids here. */
    val labels: StateFlow<Map<Int, String>> = _labels.asStateFlow()

    /** Display name for [id], falling back to the default when not yet renamed. */
    fun displayName(id: Int): String = labels.value[id] ?: defaultDisplayName(id)

    /** Set a single speaker's display name. */
    fun rename(id: Int, name: String) {
        _labels.value = _labels.value + (id to name)
    }

    /**
     * Set many display names at once. Useful for the eventual LLM-driven naming pass:
     * collect the full transcript with `Speaker 1: …\nSpeaker 2: …`, ask the LLM to
     * guess names, plug the resulting `{0: "Alice", 1: "Bob"}` map in here.
     */
    fun bulkRename(names: Map<Int, String>) {
        _labels.value = _labels.value + names
    }

    /** Clear all renames. Called at session boundaries to mirror [SpeakerRegistry.reset]. */
    fun reset() {
        _labels.value = emptyMap()
    }

    companion object {
        fun defaultDisplayName(id: Int): String = "Speaker ${id + 1}"
    }
}
