package dev.soupslurpr.transcribro.models

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import kotlin.coroutines.coroutineContext

/**
 * Streams a [ModelEntry] from its [ModelEntry.url] to local storage, verifies the
 * SHA-256, then atomically promotes the `.part` file to its final name and writes a
 * `sha256` sidecar so [InstalledModelsRepository] treats it as installed.
 *
 * The returned [Flow] emits [InstallProgress] events and finishes with either
 * [InstallProgress.Done] or [InstallProgress.Failed]. Cancellation is cooperative —
 * cancelling the collecting coroutine aborts the download and removes the partial file.
 *
 * Trust model: a download is only accepted if its SHA-256 matches [ModelEntry.sha256].
 * Mismatches imply either upstream changed under us (update the pin) or a network
 * intermediary is rewriting bytes — both should be treated as install failures, not
 * "trust on first install".
 */
class ModelInstaller(private val installed: InstalledModelsRepository) {

    fun install(entry: ModelEntry): Flow<InstallProgress> = flow {
        val dir = installed.dirFor(entry)
        dir.mkdirs()
        val finalFile = installed.modelFileIn(dir)
        val partFile = installed.partFileIn(dir)
        val shaSidecar = installed.shaSidecarIn(dir)

        // Already installed — short-circuit. Caller will see Done immediately. This makes
        // "install this if needed" callable as a no-op on already-present models.
        if (finalFile.exists() && shaSidecar.exists() && finalFile.length() > 0L) {
            Log.i(TAG, "install: ${entry.id} already present, skipping")
            emit(InstallProgress.Done(finalFile, finalFile.length()))
            return@flow
        }

        // Wipe any stale partial from a previous aborted run. Resuming a partial is a
        // separate feature — for now, restart cleanly.
        if (partFile.exists()) partFile.delete()

        emit(InstallProgress.Connecting)

        val connection = try {
            (URL(entry.url).openConnection() as HttpURLConnection).apply {
                connectTimeout = 30_000
                readTimeout = 30_000
                instanceFollowRedirects = true
                // Some CDN endpoints (HuggingFace LFS, GitHub release attachments) reject
                // or rate-limit requests with the default Java/Android User-Agent. Setting
                // an explicit UA also makes it obvious in their logs what's calling.
                setRequestProperty("User-Agent", "Stenosis/1 (Android)")
            }
        } catch (e: Exception) {
            Log.e(TAG, "openConnection failed for ${entry.url}", e)
            emit(InstallProgress.Failed("Failed to open connection: ${e.describe()}", e))
            return@flow
        }

        try {
            val responseCode = try {
                connection.responseCode
            } catch (e: Exception) {
                Log.e(TAG, "responseCode threw for ${entry.url}", e)
                emit(InstallProgress.Failed("Network error: ${e.describe()}", e))
                return@flow
            }
            if (responseCode !in 200..299) {
                emit(
                    InstallProgress.Failed(
                        "HTTP $responseCode from ${entry.url}",
                        cause = IOException("HTTP $responseCode"),
                    )
                )
                return@flow
            }

            val totalBytesHint = connection.contentLengthLong.takeIf { it > 0L }
                ?: entry.sizeBytes.takeIf { it > 0L }

            val digest = MessageDigest.getInstance("SHA-256")
            var bytesDownloaded = 0L
            emit(InstallProgress.Downloading(0L, totalBytesHint))

            try {
                connection.inputStream.use { input ->
                    partFile.outputStream().use { output ->
                        val buf = ByteArray(64 * 1024)
                        var lastEmittedAt = System.currentTimeMillis()
                        while (true) {
                            coroutineContext.ensureActive()
                            val n = input.read(buf)
                            if (n <= 0) break
                            output.write(buf, 0, n)
                            digest.update(buf, 0, n)
                            bytesDownloaded += n
                            // Throttle progress emissions — every ~100ms is plenty for UI.
                            val now = System.currentTimeMillis()
                            if (now - lastEmittedAt >= 100) {
                                emit(InstallProgress.Downloading(bytesDownloaded, totalBytesHint))
                                lastEmittedAt = now
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "stream copy failed after $bytesDownloaded bytes for ${entry.url}", e)
                partFile.delete()
                emit(
                    InstallProgress.Failed(
                        "Download interrupted after $bytesDownloaded bytes: ${e.describe()}",
                        e,
                    )
                )
                return@flow
            }

            emit(InstallProgress.Verifying)
            val actualSha = digest.digest().toHexLower()
            if (actualSha != entry.sha256) {
                partFile.delete()
                emit(
                    InstallProgress.Failed(
                        "SHA-256 mismatch — expected ${entry.sha256}, got $actualSha. " +
                            "Either the upstream pin is stale or the response was tampered with.",
                        cause = IOException("checksum mismatch"),
                    )
                )
                return@flow
            }

            if (!partFile.renameTo(finalFile)) {
                partFile.delete()
                emit(
                    InstallProgress.Failed(
                        "Failed to rename ${partFile.name} → ${finalFile.name}",
                        cause = IOException("rename failed"),
                    )
                )
                return@flow
            }
            shaSidecar.writeText(actualSha)
            installed.refresh()
            Log.i(TAG, "install: ${entry.id} ok (${finalFile.length()} bytes)")
            emit(InstallProgress.Done(finalFile, finalFile.length()))
        } finally {
            connection.disconnect()
        }
    }.flowOn(Dispatchers.IO)

    companion object {
        private const val TAG = "ModelInstaller"
    }
}

sealed class InstallProgress {
    /** Opening the HTTP connection / waiting for response headers. */
    object Connecting : InstallProgress()

    /**
     * Streaming bytes. [totalBytes] is null if the server didn't advertise a Content-Length
     * header AND the catalog entry didn't carry a size hint.
     */
    data class Downloading(val bytesDownloaded: Long, val totalBytes: Long?) : InstallProgress()

    /** Bytes are on disk; computing SHA-256 to compare against the pinned digest. */
    object Verifying : InstallProgress()

    /** Successfully installed. [file] is the final file path. */
    data class Done(val file: File, val sizeBytes: Long) : InstallProgress()

    /** Terminal failure. [message] is user-facing; [cause] is for logs. */
    data class Failed(val message: String, val cause: Throwable? = null) : InstallProgress()
}

/** Compact "ClassName: message" suitable for showing in the install error card. */
private fun Throwable.describe(): String {
    val name = this::class.simpleName ?: this::class.java.name
    val msg = message
    return if (msg.isNullOrBlank()) name else "$name: $msg"
}

private fun ByteArray.toHexLower(): String {
    val sb = StringBuilder(size * 2)
    for (b in this) {
        sb.append(HEX[(b.toInt() ushr 4) and 0x0F])
        sb.append(HEX[b.toInt() and 0x0F])
    }
    return sb.toString()
}

private val HEX = "0123456789abcdef".toCharArray()
