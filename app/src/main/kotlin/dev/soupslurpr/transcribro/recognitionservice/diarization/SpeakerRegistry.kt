package dev.soupslurpr.transcribro.recognitionservice.diarization

import android.util.Log
import kotlin.math.sqrt

/**
 * Online cosine-distance speaker clustering.
 *
 * Maintains one centroid per detected speaker as a running mean of the L2-normalized
 * embeddings assigned to it. [assign] takes a fresh embedding and either matches it to
 * an existing centroid (updating the running mean) or opens a new speaker.
 *
 * Cosine distance threshold semantics:
 *   - Embeddings produced by [PyannoteEmbedderOnnxModel] are already L2-normalized, so
 *     `1 - dot(a, b)` is cosine distance.
 *   - 0.0 = identical; 2.0 = opposite; 1.0 = orthogonal.
 *   - Empirically, wespeaker-en-voxceleb-resnet34-LM produces same-speaker pairs in the
 *     ~0.1–0.4 range and different-speaker pairs in ~0.5–1.2. [DEFAULT_THRESHOLD] is
 *     0.35: tight enough that two same-gender adult voices don't get merged on the
 *     boundary, loose enough that prosody/cadence variation in a single speaker doesn't
 *     spawn duplicates. Tune up if speakers get over-split, down if they get over-merged.
 *
 * Thread-safe: every public method takes the internal lock. The registry is shared
 * across the rolling-window pipeline (segmentation thread) and any UI consumer.
 *
 * Speaker IDs are assigned sequentially from 0 upward and are stable for the life of
 * the registry. The user-facing display name is held separately in [SpeakerLabelStore]
 * so renaming (manual or via an LLM pass after the session) doesn't touch the cluster
 * graph.
 */
class SpeakerRegistry(
    sameSpeakerThreshold: Float = DEFAULT_THRESHOLD,
) {
    private class Speaker(
        val id: Int,
        /** Sum of embeddings assigned so far. Centroid = sum / count, then L2-normalized. */
        val sum: FloatArray,
        var count: Int,
    )

    private val lock = Any()
    private val speakers = mutableListOf<Speaker>()

    /**
     * Same-speaker cosine distance threshold. Settable so a long-lived registry can
     * pick up a new value when the user adjusts it between recognition sessions,
     * without throwing away the speakers it has already learned. Reads/writes guarded
     * by [lock] alongside the speakers list.
     */
    private var sameSpeakerThreshold: Float = sameSpeakerThreshold

    /** Update the same-speaker threshold for subsequent [assign] calls. */
    fun setThreshold(threshold: Float) = synchronized(lock) {
        sameSpeakerThreshold = threshold
    }

    /**
     * Assign [embedding] (expected L2-normalized) to the closest existing speaker if its
     * cosine distance is < [sameSpeakerThreshold], otherwise open a new speaker.
     * Returns the (global) speaker id.
     */
    fun assign(embedding: FloatArray): Int = synchronized(lock) {
        require(embedding.isNotEmpty()) { "embedding must not be empty" }

        // First-ever embedding ⇒ open speaker 0.
        if (speakers.isEmpty()) {
            speakers += Speaker(id = 0, sum = embedding.copyOf(), count = 1)
            Log.i(TAG, "assign: first embedding → opened speaker 0")
            return 0
        }

        // Collect distance to each existing centroid for diagnostics + matching.
        val distances = FloatArray(speakers.size)
        var bestId = -1
        var bestDist = Float.MAX_VALUE
        for ((idx, sp) in speakers.withIndex()) {
            val centroid = computeCentroid(sp)
            val dist = cosineDistance(centroid, embedding)
            distances[idx] = dist
            if (dist < bestDist) {
                bestDist = dist
                bestId = sp.id
            }
        }

        val distSummary = speakers.indices.joinToString(", ") { i ->
            "id=${speakers[i].id}:${"%.3f".format(distances[i])}"
        }

        if (bestDist < sameSpeakerThreshold) {
            val sp = speakers.first { it.id == bestId }
            for (i in sp.sum.indices) sp.sum[i] += embedding[i]
            sp.count += 1
            Log.i(
                TAG,
                "assign: matched speaker $bestId (dist=${"%.3f".format(bestDist)} < " +
                    "${"%.3f".format(sameSpeakerThreshold)}); distances=[$distSummary]"
            )
            return bestId
        }

        val newId = speakers.size
        speakers += Speaker(id = newId, sum = embedding.copyOf(), count = 1)
        Log.i(
            TAG,
            "assign: opened new speaker $newId (best dist=${"%.3f".format(bestDist)} ≥ " +
                "${"%.3f".format(sameSpeakerThreshold)}); distances=[$distSummary]"
        )
        return newId
    }

    /** Number of speakers currently registered. */
    val size: Int get() = synchronized(lock) { speakers.size }

    /** Drop all speakers; next [assign] starts a fresh session. */
    fun reset() = synchronized(lock) {
        speakers.clear()
    }

    /**
     * Snapshot of current speaker centroids (L2-normalized) keyed by id, for diagnostics.
     * Returned arrays are defensive copies — modifying them won't affect the registry.
     */
    fun snapshotCentroids(): Map<Int, FloatArray> = synchronized(lock) {
        speakers.associate { it.id to computeCentroid(it) }
    }

    private fun computeCentroid(sp: Speaker): FloatArray {
        val mean = FloatArray(sp.sum.size) { sp.sum[it] / sp.count }
        var sumSq = 0.0
        for (v in mean) sumSq += v * v
        val norm = sqrt(sumSq).toFloat()
        if (norm == 0f) return mean
        for (i in mean.indices) mean[i] /= norm
        return mean
    }

    companion object {
        private const val TAG = "SpeakerRegistry"

        const val DEFAULT_THRESHOLD: Float = 0.35f

        /** Cosine distance between two equal-length vectors. */
        fun cosineDistance(a: FloatArray, b: FloatArray): Float {
            require(a.size == b.size) { "vector dim mismatch: ${a.size} vs ${b.size}" }
            var dot = 0.0
            var na = 0.0
            var nb = 0.0
            for (i in a.indices) {
                dot += a[i] * b[i]
                na += a[i] * a[i]
                nb += b[i] * b[i]
            }
            val denom = sqrt(na) * sqrt(nb)
            if (denom == 0.0) return 1f
            return (1.0 - dot / denom).toFloat()
        }
    }
}
