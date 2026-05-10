import java.net.URL
import java.security.MessageDigest

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose") version "2.2.0"
}

/**
 * Downloads model files into [outputDir] at relative paths given by [sources] (relPath → URL).
 * After download (or on every run for cached files), verifies SHA-256 against [checksums]
 * (relPath → lowercase hex). A mismatch deletes the file and fails the build.
 *
 * Pinning policy:
 *   - URLs reference an immutable commit / release tag, never `main`/`master`.
 *   - Checksums let us detect "the upstream silently re-published the same URL" or
 *     "a transparent CDN/mirror handed us a different file". Combined, the build either
 *     produces the exact pinned bits or fails fast.
 *
 * To bump a pin: change the URL and the checksum in lock-step. The task input map
 * change invalidates Gradle's up-to-date check and triggers a fresh download.
 */
abstract class DownloadModelsTask : DefaultTask() {
    @get:Input
    abstract val sources: MapProperty<String, String>

    @get:Input
    abstract val checksums: MapProperty<String, String>

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @TaskAction
    fun run() {
        val dir = outputDir.get().asFile
        val checks = checksums.get()
        for ((relPath, url) in sources.get()) {
            val expectedSha = checks[relPath]
                ?: throw GradleException("Missing SHA-256 for $relPath in checksums map")
            val outFile = File(dir, relPath)
            outFile.parentFile.mkdirs()

            // If a cached file is present, trust it only if its hash matches the pin.
            // A mismatch means upstream changed under us — re-download to recover.
            if (outFile.exists() && outFile.length() > 0L) {
                val cachedSha = sha256(outFile)
                if (cachedSha == expectedSha) {
                    logger.lifecycle("[downloadModels] $relPath cached ${outFile.length()} bytes (sha verified)")
                    continue
                }
                logger.lifecycle("[downloadModels] $relPath cached but sha mismatch (got $cachedSha, want $expectedSha) — redownloading")
                outFile.delete()
            }

            logger.lifecycle("[downloadModels] downloading $relPath")
            logger.lifecycle("[downloadModels]   from $url")
            val tmp = File(outFile.parentFile, "${outFile.name}.part")
            try {
                URL(url).openStream().use { input ->
                    tmp.outputStream().use { output -> input.copyTo(output) }
                }
                val actualSha = sha256(tmp)
                if (actualSha != expectedSha) {
                    tmp.delete()
                    throw GradleException(
                        "Checksum mismatch for $relPath\n" +
                            "  url:      $url\n" +
                            "  expected: $expectedSha\n" +
                            "  actual:   $actualSha\n" +
                            "Either upstream changed (update the pin) or a mirror/proxy is " +
                            "tampering with the response."
                    )
                }
                if (!tmp.renameTo(outFile)) {
                    error("Failed to move ${tmp.absolutePath} to ${outFile.absolutePath}")
                }
                logger.lifecycle("[downloadModels]   ok ${outFile.length()} bytes (sha verified)")
            } catch (e: GradleException) {
                throw e
            } catch (e: Exception) {
                tmp.delete()
                throw GradleException(
                    "Failed to download $url (${e::class.simpleName}: ${e.message})",
                    e,
                )
            }
        }
    }

    private fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(8192)
            while (true) {
                val n = input.read(buf)
                if (n <= 0) break
                md.update(buf, 0, n)
            }
        }
        val bytes = md.digest()
        val sb = StringBuilder(bytes.size * 2)
        for (b in bytes) {
            sb.append(String.format("%02x", b.toInt() and 0xff))
        }
        return sb.toString()
    }
}

// Every model checkpoint pinned to an immutable URL + verified by SHA-256. A bump means
// editing the URL AND the checksum in lock-step — the mismatch surfaces in the build
// output if you forget. URLs and SHAs are sister entries keyed by the same relative path.
val downloadModels by tasks.registering(DownloadModelsTask::class) {
    description = "Downloads all bundled ONNX/GGML models into the cache dir."
    group = "build"

    // whisper.cpp transcription model (tiny.en, q8_0 quantization).
    sources.put(
        "models/whisper/ggml-model-whisper-tiny.en-q8_0.bin",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny.en-q8_0.bin",
    )
    checksums.put(
        "models/whisper/ggml-model-whisper-tiny.en-q8_0.bin",
        "5bc2b3860aa151a4c6e7bb095e1fcce7cf12c7b020ca08dcec0c6d018bb7dd94",
    )

    // SileroVAD v5 voice-activity-detection (state[2,batch,128] signature — the wrapper
    // in SileroVadOnnxModel must match this version's I/O exactly; bumping this without
    // updating the wrapper code will surface as a schema-mismatch error at runtime).
    sources.put(
        "models/silero_vad/silero_vad.onnx",
        "https://github.com/snakers4/silero-vad/raw/bfdc0193023f121ea5b3cc7b176dbed570a68a59/src/silero_vad/data/silero_vad.onnx",
    )
    checksums.put(
        "models/silero_vad/silero_vad.onnx",
        "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
    )

    // pyannote v3 speaker segmentation (sherpa-onnx export with input "x" / output "y").
    sources.put(
        "models/pyannote/segmentation-3.0.onnx",
        "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/9403a6902bb58e3d5ae8c7e77c3422de279db2e0/model.onnx",
    )
    checksums.put(
        "models/pyannote/segmentation-3.0.onnx",
        "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
    )

    // wespeaker speaker-embedding (sherpa-onnx export, "feats" → "embs"). URL is already
    // pinned via release tag, but we still verify SHA-256 like the others.
    sources.put(
        "models/pyannote/wespeaker-en-voxceleb-resnet34-LM.onnx",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx",
    )
    checksums.put(
        "models/pyannote/wespeaker-en-voxceleb-resnet34-LM.onnx",
        "e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012",
    )

    // Cache outside the project tree so `gradlew clean` doesn't force a ~70MB re-download
    // and so multiple worktrees on the same machine share the downloads. Gradle user home
    // is ~/.gradle by default. AGP follows whatever the DirectoryProperty resolves to,
    // and the path doesn't have to exist at config time — the task creates it on first run.
    outputDir.set(
        project.layout.dir(
            provider {
                File(gradle.gradleUserHomeDir, "transcribro-model-cache")
            },
        ),
    )
}

android {
    namespace = "dev.soupslurpr.transcribro"
    compileSdk = 36
    buildToolsVersion = "36.0.0"
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "dev.soupslurpr.transcribro"
        minSdk = 29
        targetSdk = 36
        versionCode = 7
        versionName = versionCode.toString()

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }

        ndk {
            abiFilters.addAll(setOf("arm64-v8a", "x86_64"))
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
    androidResources {
        generateLocaleConfig = true
        localeFilters += listOf("en")
    }
    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        getByName("debug") {
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-debug"
            signingConfig = signingConfigs.getByName("debug")
        }
        create("staging") {
            initWith(getByName("release"))
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-debug"
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    dependenciesInfo {
        // Disables dependency metadata when building APKs.
        includeInApk = false
        // Disables dependency metadata when building Android App Bundles.
        includeInBundle = false
    }
}

// Wire downloadModels' output directory in as a generated assets source. The merge-assets
// step depends on it automatically, so a fresh build downloads any missing models and a
// subsequent build is up-to-date.
androidComponents {
    onVariants { variant ->
        variant.sources.assets?.addGeneratedSourceDirectory(
            downloadModels,
            DownloadModelsTask::outputDir,
        )
    }
}

dependencies {

    implementation(project(":lib"))

    implementation("androidx.core:core-ktx:1.16.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.9.2")
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.navigation:navigation-compose:2.9.3")
    implementation("androidx.activity:activity-ktx:1.10.1")

    implementation(platform("androidx.compose:compose-bom:2025.07.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-core")
    implementation("androidx.compose.material:material-icons-extended")
    debugImplementation("androidx.compose.ui:ui-tooling")

    implementation("androidx.datastore:datastore-preferences:1.1.7")
    implementation("com.google.accompanist:accompanist-permissions:0.37.3")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")
}