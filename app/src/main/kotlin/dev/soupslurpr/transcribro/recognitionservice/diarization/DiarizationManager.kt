package dev.soupslurpr.transcribro.recognitionservice.diarization

import android.util.Log
import dev.soupslurpr.transcribro.recognitionservice.diarization.PyannoteSegmenterOnnxModel.Companion.NUM_FRAMES
import dev.soupslurpr.transcribro.recognitionservice.diarization.PyannoteSegmenterOnnxModel.Companion.NUM_LOCAL_SPEAKERS
import dev.soupslurpr.transcribro.recognitionservice.diarization.PyannoteSegmenterOnnxModel.Companion.WINDOW_SAMPLES

/** Closed per-speaker segment ready for high-quality transcription downstream. */
data class SpeakerSegmentEvent(
    val globalSpeakerId: Int,
    /** Speaker's audio for this segment, with other-speaker spans replaced by silence (0). */
    val audio: ShortArray,
    /** Absolute sample index (since session start) of the first active sample in this segment. */
    val sampleRangeStart: Long,
    /** Absolute sample index (exclusive) of the last active sample. */
    val sampleRangeEnd: Long,
)

/**
 * Real-time speaker diarization driven by pyannote-segmentation on a rolling 10s window.
 *
 * Per audio buffer ([process]):
 *   1. Append to the rolling 10s window (drop oldest).
 *   2. Once per ~1s of new audio, run segmentation. Pyannote local ids 0/1/2 within the
 *      window are mapped to stable global ids by embedding each local speaker's longest
 *      contiguous turn (≥[minEmbeddingDurationSamples]) and feeding that to [registry].
 *   3. For the new ~1s portion, append the actual sample to the active speakers' tracks
 *      and silence (0) to the open-segment tracks of inactive speakers.
 *   4. When a speaker has been silent on their track for ≥[silenceThresholdSamples],
 *      close the open segment and emit a [SpeakerSegmentEvent] via [onSegmentClosed].
 *
 * Per-speaker tracks are NOT clean source-separated audio — they're the original
 * single-channel mic input with non-this-speaker spans zeroed out. For non-overlapping
 * speech this works well; for overlapping speech, both speakers' tracks contain the
 * overlap with the other speaker as background noise. The interface here ([process],
 * event emissions) is shaped so a SepFormer-grade source separator could later replace
 * the audio splitter without changes downstream.
 *
 * Not thread-safe: caller serializes [process] / [flush] / [reset]. Typically the
 * recording-loop coroutine drives [process] one buffer at a time, [onSegmentClosed]
 * fires synchronously during [process], and the consumer dispatches whisper transcribe
 * jobs to its own scope.
 */
class DiarizationManager(
    private val segmenter: PyannoteSegmenterOnnxModel,
    private val embedder: PyannoteEmbedderOnnxModel,
    private val registry: SpeakerRegistry,
    private val onSegmentClosed: (SpeakerSegmentEvent) -> Unit,
    sampleRate: Int = SAMPLE_RATE,
    silenceMs: Int = 3000,
    segmentationStrideMs: Int = 1000,
    // Lowered from 1500 → 700 → 500: in real conversations, individual turns are
    // routinely under 1s. Setting the threshold too high means embedding never runs
    // and the diarization manager never assigns global speaker ids to anyone. 500ms
    // matches the embedder's hard minimum (PyannoteEmbedderOnnxModel.MIN_AUDIO_SAMPLES)
    // — embeddings get noisier at this length, but a noisy embedding is still better
    // than none, and the registry's cosine threshold absorbs some of the wobble.
    minEmbeddingDurationMs: Int = 500,
) {
    private val silenceThresholdSamples: Int = sampleRate * silenceMs / 1000
    private val segmentationStrideSamples: Int = sampleRate * segmentationStrideMs / 1000
    private val minEmbeddingDurationSamples: Int = sampleRate * minEmbeddingDurationMs / 1000

    private class SpeakerTrack(val globalId: Int) {
        // ArrayDeque so dropping consumed prefix after emit is O(1) per element.
        val audio: ArrayDeque<Short> = ArrayDeque()
        var lastActiveSample: Long = -1L
        var segmentStartSample: Long = -1L
        /** Index into [audio] where the current segment begins. */
        var segmentStartIndex: Int = -1
    }

    private val window = ShortArray(WINDOW_SAMPLES)
    private var windowSize = 0
    private var samplesSeen: Long = 0
    private var nextSegmentationAt: Long = -1L  // absolute sample at which next stride should run

    // Cache the local→global mapping by which local-speaker-slots have any activity
    // in the stride. As long as that set doesn't change, the slots correspond to the
    // same physical speakers and we can skip the (expensive) wespeaker embedding pass.
    // Recomputed whenever the active-set bitmask changes.
    private var cachedMappingKey: Int = -1
    private var cachedMapping: Map<Int, Int> = emptyMap()

    private val tracks = mutableMapOf<Int, SpeakerTrack>()

    // Reused per-segmentation buffer — saves a 160k-float alloc per second.
    private val windowFloatBuf = FloatArray(WINDOW_SAMPLES)

    /**
     * Feed one chunk of int16 audio. Triggers segmentation passes as enough time accumulates;
     * may invoke [onSegmentClosed] zero or more times before returning.
     */
    fun process(buffer: ShortArray) {
        if (buffer.isEmpty()) return

        appendToWindow(buffer)
        samplesSeen += buffer.size

        if (windowSize < WINDOW_SAMPLES) return

        // Initialise the segmentation cursor on the first full window — start processing
        // from the beginning of that window.
        if (nextSegmentationAt < 0) {
            nextSegmentationAt = samplesSeen - WINDOW_SAMPLES + segmentationStrideSamples.toLong()
        }

        // Drain any strides we owe.
        while (samplesSeen >= nextSegmentationAt) {
            val portionEnd = nextSegmentationAt
            val portionStart = portionEnd - segmentationStrideSamples
            runSegmentationStride(portionStart, portionEnd)
            nextSegmentationAt += segmentationStrideSamples
        }

        checkSegmentClosures()
    }

    /**
     * Force-close any open segments. Call when recognition is ending.
     *
     * Crucially, this also handles the "recording was shorter than the 10s window"
     * case: pyannote-segmentation requires exactly 10s of input, so a 5s recording
     * never triggers a segmentation pass during the audio loop. Here we pad the
     * window with leading silence to 10s and run one final pass over just the
     * actual-audio portion, so short recordings still produce per-speaker segments.
     */
    fun flush() {
        if (windowSize in 1 until WINDOW_SAMPLES) {
            // Shift the audio we have to the END of the window and zero-pad the front.
            val realAudioSamples = windowSize
            val padding = WINDOW_SAMPLES - realAudioSamples
            System.arraycopy(window, 0, window, padding, realAudioSamples)
            for (i in 0 until padding) window[i] = 0
            windowSize = WINDOW_SAMPLES
            // The audio actually covers (samplesSeen - realAudioSamples, samplesSeen].
            runSegmentationStride(samplesSeen - realAudioSamples, samplesSeen)
        } else if (windowSize == WINDOW_SAMPLES &&
                   nextSegmentationAt in (samplesSeen - segmentationStrideSamples + 1)..samplesSeen
        ) {
            // Full window, but there's a partial stride that the audio loop didn't
            // get to process yet.
            val portionStart = nextSegmentationAt - segmentationStrideSamples
            runSegmentationStride(portionStart, samplesSeen)
        }
        for (track in tracks.values.toList()) {
            if (track.segmentStartSample >= 0) {
                emitClosed(track)
            }
        }
    }

    /**
     * Snapshot the rolling audio window for an external consumer (typically the
     * streaming-preview pass running alongside diarization). Returns a defensive copy.
     */
    fun snapshotWindow(): ShortArray {
        if (windowSize == 0) return ShortArray(0)
        return window.copyOf(windowSize)
    }

    /** Cumulative sample count since [reset] (or session start). */
    fun samplesSeen(): Long = samplesSeen

    /** Reset all state for a new session. Also resets the speaker [registry]. */
    fun reset() {
        windowSize = 0
        samplesSeen = 0
        nextSegmentationAt = -1L
        tracks.clear()
        cachedMappingKey = -1
        cachedMapping = emptyMap()
        registry.reset()
    }

    private fun appendToWindow(buffer: ShortArray) {
        val n = buffer.size
        if (windowSize < WINDOW_SAMPLES) {
            val toCopy = minOf(n, WINDOW_SAMPLES - windowSize)
            System.arraycopy(buffer, 0, window, windowSize, toCopy)
            windowSize += toCopy
            if (toCopy < n) {
                slideWindow(buffer, toCopy, n - toCopy)
            }
        } else {
            slideWindow(buffer, 0, n)
        }
    }

    private fun slideWindow(src: ShortArray, srcOffset: Int, count: Int) {
        if (count >= WINDOW_SAMPLES) {
            System.arraycopy(src, srcOffset + count - WINDOW_SAMPLES, window, 0, WINDOW_SAMPLES)
        } else {
            System.arraycopy(window, count, window, 0, WINDOW_SAMPLES - count)
            System.arraycopy(src, srcOffset, window, WINDOW_SAMPLES - count, count)
        }
    }

    private fun runSegmentationStride(portionStartSample: Long, portionEndSample: Long) {
        for (i in 0 until WINDOW_SAMPLES) {
            windowFloatBuf[i] = window[i] / 32768f
        }
        val segResult = segmenter.segment(windowFloatBuf)

        // Diagnostic: how many frames each local speaker was active for in this stride's
        // window. Lets us tell whether pyannote actually heard a second voice or whether
        // an apparent miss is an audio-level issue (zeros across the board).
        val perSpeakerActiveFrames = IntArray(segResult.numLocalSpeakers)
        for (frame in 0 until segResult.numFrames) {
            for (sp in 0 until segResult.numLocalSpeakers) {
                if (segResult.activity[frame][sp]) perSpeakerActiveFrames[sp]++
            }
        }
        // Also dump the powerset class histogram. If pyannote only ever picks class 0
        // (silence) and one other class, the model is treating the whole window as one
        // speaker. If it picks classes for slots 0/2 but our active-frame-counts say
        // those slots are empty, the bug is in our decoding, not the model.
        val classHist = segResult.classHistogram
        Log.i(
            TAG,
            "segmentation stride [$portionStartSample..$portionEndSample): " +
                "active-frame-counts=${perSpeakerActiveFrames.toList()} / ${segResult.numFrames}, " +
                "class-hist=${classHist.toList()} (silent,{0},{1},{2},{0,1},{0,2},{1,2})"
        )

        val mapping = computeLocalToGlobalMapping(segResult)
        if (mapping.isNotEmpty()) {
            Log.i(TAG, "local→global mapping for this stride: $mapping")
        }

        // Update per-speaker tracks for the new portion of audio.
        val windowStartSample = samplesSeen - windowSize
        for (absSample in portionStartSample until portionEndSample) {
            val windowIdx = (absSample - windowStartSample).toInt()
            if (windowIdx !in 0 until windowSize) continue
            val frameIdx = (windowIdx.toLong() * segResult.numFrames / WINDOW_SAMPLES).toInt()
                .coerceIn(0, segResult.numFrames - 1)
            val sampleValue = window[windowIdx]

            val activeGlobals = mutableSetOf<Int>()
            val activity = segResult.activity[frameIdx]
            for (local in activity.indices) {
                if (activity[local]) {
                    val gid = mapping[local] ?: continue
                    activeGlobals.add(gid)
                }
            }

            // Active globals: append the actual sample. Open a segment if not already open.
            for (gid in activeGlobals) {
                val track = tracks.getOrPut(gid) { SpeakerTrack(gid) }
                track.audio.addLast(sampleValue)
                track.lastActiveSample = absSample
                if (track.segmentStartSample < 0) {
                    track.segmentStartSample = absSample
                    track.segmentStartIndex = track.audio.size - 1
                }
            }
            // Inactive globals with an open segment: append silence so the track stays
            // sample-aligned for whisper. Closed-segment tracks don't accumulate silence.
            for ((gid, track) in tracks) {
                if (gid !in activeGlobals && track.segmentStartSample >= 0) {
                    track.audio.addLast(0)
                }
            }
        }
    }

    /**
     * For each local speaker, find their longest contiguous active turn in the segmentation
     * window; if it's long enough to embed reliably, embed it and assign a global id.
     * Local speakers without a long-enough turn aren't mapped — their frames effectively
     * count as silence for everyone in this stride.
     *
     * Caches the result by "which local-speaker-slots have any activity" — when that
     * bitmask is unchanged from the previous stride AND every active slot already has
     * a global id, the slot↔speaker correspondence inside pyannote-segmentation is
     * stable across the window and we skip the expensive wespeaker embedding pass.
     * If an active slot is still unmapped (its longest turn was below threshold last
     * stride), we keep retrying it — turns often grow as the rolling window slides
     * across more of the speaker's utterance. This is the difference between running
     * embedding once per stride (~260ms) vs running it once per speaker change, which
     * keeps the audio loop able to consume real-time-rate input.
     */
    private fun computeLocalToGlobalMapping(seg: SegmentationResult): Map<Int, Int> {
        // Bitmask of which local speakers have any active frame in this stride's window.
        var activityKey = 0
        for (localSp in 0 until seg.numLocalSpeakers) {
            for (frame in 0 until seg.numFrames) {
                if (seg.activity[frame][localSp]) {
                    activityKey = activityKey or (1 shl localSp)
                    break
                }
            }
        }
        // The cache short-circuits embedding only if the same set of slots is active AND
        // every active slot already has a global id. Otherwise we keep retrying the
        // unmapped slots — their longest contiguous turn may grow as the rolling window
        // slides across more of the speaker's utterance, eventually clearing the
        // [minEmbeddingDurationSamples] threshold.
        val allActiveSlotsMapped = (0 until seg.numLocalSpeakers).all { localSp ->
            ((activityKey shr localSp) and 1) == 0 || cachedMapping.containsKey(localSp)
        }
        if (activityKey == cachedMappingKey && allActiveSlotsMapped && cachedMapping.isNotEmpty()) {
            return cachedMapping
        }

        val result = mutableMapOf<Int, Int>()
        for (localSp in 0 until seg.numLocalSpeakers) {
            if ((activityKey and (1 shl localSp)) == 0) continue
            val turn = findLongestTurn(seg, localSp) ?: continue
            val turnStart = (turn.startFrame.toLong() * WINDOW_SAMPLES / seg.numFrames).toInt()
            val turnEnd = ((turn.endFrame + 1).toLong() * WINDOW_SAMPLES / seg.numFrames).toInt()
            val durationSamples = turnEnd - turnStart
            if (durationSamples < minEmbeddingDurationSamples) {
                Log.i(
                    TAG,
                    "skip embedding for local $localSp: longest turn $durationSamples samples " +
                        "(${"%.2f".format(durationSamples / SAMPLE_RATE.toDouble())}s) " +
                        "< minimum $minEmbeddingDurationSamples"
                )
                continue
            }

            // Extract turn audio in int16 amplitude range (NOT [-1, 1]) for KaldiFbank.
            val turnAudio = FloatArray(durationSamples) { i ->
                window[turnStart + i].toFloat()
            }
            val embedding = embedder.embed(turnAudio)
            val globalId = registry.assign(embedding)
            Log.i(
                TAG,
                "embedded local $localSp turn (${"%.2f".format(durationSamples / SAMPLE_RATE.toDouble())}s) -> global $globalId " +
                    "(registry size=${registry.size})"
            )
            result[localSp] = globalId
        }

        cachedMappingKey = activityKey
        cachedMapping = result
        return result
    }

    private fun checkSegmentClosures() {
        // Snapshot to avoid concurrent-modification when emitClosed clears tracks.
        val snapshot = tracks.values.toList()
        for (track in snapshot) {
            if (track.segmentStartSample < 0) continue
            if (samplesSeen - track.lastActiveSample >= silenceThresholdSamples) {
                emitClosed(track)
            }
        }
    }

    private fun emitClosed(track: SpeakerTrack) {
        val startIdx = track.segmentStartIndex
        // The last index that contains an active sample (rather than trailing silence-fill).
        val activeSpan = (track.lastActiveSample - track.segmentStartSample).toInt()
        val endIdx = startIdx + activeSpan
        if (endIdx < startIdx || endIdx >= track.audio.size) {
            // Defensive: shouldn't happen, but bail cleanly.
            track.segmentStartSample = -1L
            track.segmentStartIndex = -1
            track.audio.clear()
            return
        }

        val audio = ShortArray(endIdx - startIdx + 1) { track.audio[startIdx + it] }
        val event = SpeakerSegmentEvent(
            globalSpeakerId = track.globalId,
            audio = audio,
            sampleRangeStart = track.segmentStartSample,
            sampleRangeEnd = track.lastActiveSample + 1,
        )
        Log.i(
            TAG,
            "speaker ${track.globalId} segment closed: samples=${audio.size} " +
                "range=[${event.sampleRangeStart}..${event.sampleRangeEnd})"
        )
        onSegmentClosed(event)

        // Drop everything; segment is done. Next time this speaker becomes active they'll
        // start with a fresh empty buffer.
        track.audio.clear()
        track.segmentStartSample = -1L
        track.segmentStartIndex = -1
    }

    private data class TurnRange(val startFrame: Int, val endFrame: Int) {
        val duration: Int get() = endFrame - startFrame + 1
    }

    /**
     * Find the longest contiguous run where [localSp] is the **exclusive** active speaker
     * (powerset classes 1/2/3 — not the overlap classes 4/5/6). Used to extract clean
     * single-speaker audio for the embedder; including overlap frames mixes both voices'
     * acoustic content into one chunk, which after CMVN drifts the embedding off the real
     * speaker centroid by 0.4+ cosine distance and spawns phantom speakers.
     *
     * Per-sample track appending still uses the inclusive activity matrix — overlap
     * regions get appended to both speakers' tracks (with the other voice as background)
     * so the audio splitter stays sample-aligned.
     */
    private fun findLongestTurn(seg: SegmentationResult, localSp: Int): TurnRange? {
        var best: TurnRange? = null
        var currentStart = -1
        for (i in 0 until seg.numFrames) {
            val active = seg.isExclusiveAt(i, localSp)
            if (active) {
                if (currentStart < 0) currentStart = i
            } else if (currentStart >= 0) {
                val candidate = TurnRange(currentStart, i - 1)
                if (best == null || candidate.duration > best.duration) best = candidate
                currentStart = -1
            }
        }
        if (currentStart >= 0) {
            val candidate = TurnRange(currentStart, seg.numFrames - 1)
            if (best == null || candidate.duration > best.duration) best = candidate
        }
        return best
    }

    @Suppress("unused") // exposed for diagnostics / tests later
    fun openSegmentCount(): Int = tracks.values.count { it.segmentStartSample >= 0 }

    companion object {
        const val SAMPLE_RATE: Int = 16000
        private const val TAG = "DiarizationManager"

        // Re-export the underlying model constants so callers don't have to import three packages.
        const val WINDOW_SAMPLES_PUBLIC: Int = WINDOW_SAMPLES
        const val NUM_FRAMES_PUBLIC: Int = NUM_FRAMES
        const val NUM_LOCAL_SPEAKERS_PUBLIC: Int = NUM_LOCAL_SPEAKERS
    }
}
