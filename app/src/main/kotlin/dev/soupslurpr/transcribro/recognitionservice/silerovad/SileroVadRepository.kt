package dev.soupslurpr.transcribro.recognitionservice.silerovad

import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf

class SileroVadRepository(
    private val sileroVadLocalDataSource: SileroVadLocalDataSource
) {

    private var sileroVadDetector: MutableState<SileroVadDetector?> =
        mutableStateOf(null)

    private suspend fun loadSileroVadDetectorIfNull() {
        if (sileroVadDetector.value == null) {
            sileroVadDetector.value = sileroVadLocalDataSource.getSileroVadDetector()
        }
    }

    /** Eagerly load + schema-verify the model so first-use latency and failure modes
     *  are exposed at service create time rather than mid-recognition. */
    suspend fun preload() {
        loadSileroVadDetectorIfNull()
    }

    suspend fun detect(data: ShortArray): Map<String, Double>? {
        loadSileroVadDetectorIfNull()

        val buffer = FloatArray(data.size) { index ->
            (data[index] / 32767.0f).coerceIn(-1f..1f)
        }

        val detectResult = sileroVadDetector.value?.apply(buffer, true)

        return detectResult
    }

    fun release() {
        sileroVadDetector.value?.close()
    }

    fun reset() {
        sileroVadDetector.value?.reset()
    }
}