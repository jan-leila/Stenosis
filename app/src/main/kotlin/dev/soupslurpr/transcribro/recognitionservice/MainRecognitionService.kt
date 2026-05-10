package dev.soupslurpr.transcribro.recognitionservice

import android.content.ContextParams
import android.content.Intent
import android.media.AudioFormat
import android.media.AudioRecord
import android.os.Build
import android.os.Bundle
import android.os.RemoteException
import android.speech.RecognitionService
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import android.widget.Toast
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.whispercpp.rnnoise.Denoiser
import com.whispercpp.whisper.WhisperContext
import dev.soupslurpr.transcribro.recognitionservice.diarization.DiarizationManager
import dev.soupslurpr.transcribro.recognitionservice.diarization.DiarizationRepository
import dev.soupslurpr.transcribro.recognitionservice.diarization.PyannoteEmbedderApi
import dev.soupslurpr.transcribro.recognitionservice.diarization.PyannoteEmbedderLocalDataSource
import dev.soupslurpr.transcribro.recognitionservice.diarization.PyannoteEmbedderOnnxModel
import dev.soupslurpr.transcribro.recognitionservice.diarization.PyannoteSegmenterApi
import dev.soupslurpr.transcribro.recognitionservice.diarization.PyannoteSegmenterLocalDataSource
import dev.soupslurpr.transcribro.recognitionservice.diarization.PyannoteSegmenterOnnxModel
import dev.soupslurpr.transcribro.recognitionservice.diarization.SpeakerRegistry
import dev.soupslurpr.transcribro.recognitionservice.diarization.DiarizationAssets
import dev.soupslurpr.transcribro.recognitionservice.silerovad.SileroVadApi
import dev.soupslurpr.transcribro.recognitionservice.silerovad.SileroVadDetector
import dev.soupslurpr.transcribro.recognitionservice.silerovad.SileroVadLocalDataSource
import dev.soupslurpr.transcribro.recognitionservice.silerovad.SileroVadRepository
import dev.soupslurpr.transcribro.recognitionservice.whisper.WhisperApi
import dev.soupslurpr.transcribro.recognitionservice.whisper.WhisperLocalDataSource
import dev.soupslurpr.transcribro.recognitionservice.whisper.WhisperRepository
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import java.lang.System.currentTimeMillis
import java.util.concurrent.atomic.AtomicBoolean

private data class Transcription(
    val audioData: MutableList<Short> = mutableListOf(),
    var start: Double?,
    var end: Double?,
    var text: String?,
)

class MainRecognitionService : RecognitionService() {
    companion object {
        const val EXTRA_AUTO_STOP = "dev.soupslurpr.transcribro.EXTRA_AUTO_STOP"
        const val EXTRA_STREAMING = "dev.soupslurpr.transcribro.EXTRA_STREAMING"
        const val EXTRA_DIARIZATION = "dev.soupslurpr.transcribro.EXTRA_DIARIZATION"
        // Apply RNNoise denoising to the mic input before it reaches whisper, silero,
        // and the diarization pipeline. Defaults to true if the caller doesn't set it.
        const val EXTRA_NOISE_SUPPRESSION = "dev.soupslurpr.transcribro.EXTRA_NOISE_SUPPRESSION"
        // Speaker id (Int) attached to a partial bundle when diarization is on.
        // The display name lookup happens on the receiver via SpeakerLabelStore.
        const val RESULTS_SPEAKER_ID = "dev.soupslurpr.transcribro.RESULTS_SPEAKER_ID"
        // Marks a partialResults bundle as a live streaming preview that the IME
        // should render with setComposingText (and finalize when the segment-final partial arrives).
        const val RESULTS_STREAMING_PARTIAL = "dev.soupslurpr.transcribro.RESULTS_STREAMING_PARTIAL"
        // Re-run streaming inference once we've accumulated this many new samples (1s @ 16kHz).
        private const val STREAMING_TRIGGER_SAMPLES = 16000
        // The streaming pass only ever sees the most-recent N samples of the segment.
        // Whisper's encoder maxes out at 30s of context, and we use single_segment=true,
        // so anything older than this window would either be dropped silently or cause
        // the preview to keep echoing the start of the segment. The segment-final pass
        // (run when VAD detects end-of-speech) still sees the entire segment.
        private const val STREAMING_WINDOW_SAMPLES = 10 * 16000
        private const val TAG = "MainRecognitionService"

        // RMS amplitude (int16) below which we consider the streaming snapshot "silent"
        // and skip the whisper pass. Without this, whisper hallucinates plausible-
        // sounding nonsense on quiet audio ("you know what they..." / "♪♪") and the
        // preview displays it as if it were real speech.
        private const val STREAMING_SILENCE_RMS_THRESHOLD = 300f

        // Adaptive audio_ctx (whisper encoder context length) — smaller is faster, less accurate.
        // We tune it so the streaming partial lands roughly STREAMING_TARGET_STALENESS_MS after
        // the last sample in its snapshot was captured.
        private const val STREAMING_TARGET_STALENESS_MS = 1000L
        private const val STREAMING_AUDIO_CTX_INITIAL = 768
        private const val STREAMING_AUDIO_CTX_MIN = 256
        private const val STREAMING_AUDIO_CTX_MAX = 1500
        private const val STREAMING_AUDIO_CTX_STEP = 128

        // When the sliding window has left at least this much audio behind un-transcribed,
        // run a one-off "lock-in" whisper pass on the orphaned chunk and append its text
        // to the confirmed prefix. Larger = fewer locks but bigger preview-jumps; smaller =
        // more frequent locks (more total whisper work) but smoother prefix growth. 2s is
        // also large enough that we rarely cut a chunk in the middle of a word.
        private const val STREAMING_LOCKIN_GAP_SAMPLES = 2 * 16000
    }

    private val recordAndTranscribeScope = CoroutineScope(Dispatchers.IO)

    private val transcribeScope = CoroutineScope(Dispatchers.IO)

    private val isRecording = AtomicBoolean(true)

    private var recordAndTranscribeJob: Job? = null

//    private var loadedModel: WhichModel? = null

    private var isSpeaking by mutableStateOf(false)

    private var stopListening by mutableStateOf(false)

    private val transcribeJobs = mutableListOf<Job>()

    private val streamingJobs = mutableListOf<Job>()

    private val whisperRepository: WhisperRepository =
        WhisperRepository(
            WhisperLocalDataSource(
                whisperApi =
                object : WhisperApi {
                    override fun getWhisperContext(): WhisperContext {
                        return WhisperContext.createContextFromAsset(
                            application.assets,
                            "models/whisper/ggml-model-whisper-tiny.en-q8_0.bin"
                        )
                    }
                },
                ioDispatcher = Dispatchers.IO,
            )
        )

    private val sileroVadRepository = SileroVadRepository(
        SileroVadLocalDataSource(
            object : SileroVadApi {
                override fun getSileroVadDetector(): SileroVadDetector {
                    val SAMPLE_RATE = 16000
                    val START_THRESHOLD = 0.6f
                    val END_THRESHOLD = 0.45f
                    val MIN_SILENCE_DURATION_MS = 3000
                    val SPEECH_PAD_MS = 0

                    val model =
                        this@MainRecognitionService.assets.open("models/silero_vad/silero_vad.onnx")

                    val modelBytes = model.readBytes()

                    model.close()

                    return SileroVadDetector(
                        modelBytes,
                        START_THRESHOLD,
                        END_THRESHOLD,
                        SAMPLE_RATE,
                        MIN_SILENCE_DURATION_MS,
                        SPEECH_PAD_MS
                    )
                }
            },
            Dispatchers.IO
        )
    )

    // Diarization is opt-in (via the diarizationEnabled preference / EXTRA_DIARIZATION
    // intent extra) — its models are loaded on first use only, not at service onCreate.
    private val diarizationRepository = DiarizationRepository(
        segmenterDataSource = PyannoteSegmenterLocalDataSource(
            api = object : PyannoteSegmenterApi {
                override fun getSegmenter(): PyannoteSegmenterOnnxModel {
                    val bytes = this@MainRecognitionService.assets
                        .open(DiarizationAssets.SEGMENTATION_ASSET)
                        .use { it.readBytes() }
                    return PyannoteSegmenterOnnxModel(bytes)
                }
            },
            ioDispatcher = Dispatchers.IO,
        ),
        embedderDataSource = PyannoteEmbedderLocalDataSource(
            api = object : PyannoteEmbedderApi {
                override fun getEmbedder(): PyannoteEmbedderOnnxModel {
                    val bytes = this@MainRecognitionService.assets
                        .open(DiarizationAssets.EMBEDDER_ASSET)
                        .use { it.readBytes() }
                    return PyannoteEmbedderOnnxModel(bytes)
                }
            },
            ioDispatcher = Dispatchers.IO,
        ),
    )

    // Persistent speaker registry across recognition sessions. Reset on service teardown
    // (or, in future, on an explicit user gesture). DiarizationManager.reset() also
    // resets this registry when the manager itself is reset.
    private val speakerRegistry = SpeakerRegistry()

    /**
     * Set when a model fails to load (e.g. its ONNX I/O signature doesn't match what
     * the wrapper code was written against). When non-null, [onStartListening] short-
     * circuits with [SpeechRecognizer.ERROR_CLIENT]. The user has already seen a toast
     * with the underlying error from [onCreate].
     */
    private var modelLoadError: String? = null

    override fun onCreate() {
        super.onCreate()
        // Pre-load whisper, silero, and the diarization models so first-use latency and
        // any schema-mismatch failures surface here rather than crashing the audio loop
        // later. Diarization adds ~30MB of RAM, but doing it here means the user gets
        // immediate startup with a toast on failure (no explicit "verify" button needed).
        recordAndTranscribeScope.launch {
            try {
                whisperRepository.preload()
                Log.i(TAG, "whisper context preloaded")
                sileroVadRepository.preload()
                Log.i(TAG, "silero VAD detector preloaded")
                diarizationRepository.preload()
                Log.i(TAG, "diarization models preloaded")
            } catch (e: ModelSchemaMismatchException) {
                Log.e(TAG, "model schema mismatch during preload", e)
                modelLoadError = "${e.modelName}: ${e.details}"
                withContext(Dispatchers.Main) {
                    Toast.makeText(
                        this@MainRecognitionService,
                        "Transcribro: ${e.modelName} model is incompatible. " +
                            "Recognition disabled until the wrapper is updated.",
                        Toast.LENGTH_LONG,
                    ).show()
                }
            } catch (e: Throwable) {
                Log.e(TAG, "unexpected preload failure", e)
                modelLoadError = "${e::class.simpleName}: ${e.message}"
            }
        }
    }

    override fun onStartListening(recognizerIntent: Intent?, listener: Callback?) {
        // Bail out cleanly if a model failed to load earlier. The toast was already
        // shown once at preload time; we just refuse to start recognition.
        modelLoadError?.let { err ->
            Log.w(TAG, "onStartListening rejected: $err")
            listener?.error(SpeechRecognizer.ERROR_CLIENT)
            return
        }

        val autoStopRecognition = recognizerIntent?.extras?.getBoolean(EXTRA_AUTO_STOP) ?: true
        val isPartialResults = recognizerIntent?.extras?.getBoolean(RecognizerIntent.EXTRA_PARTIAL_RESULTS)
        val streamingEnabled = recognizerIntent?.extras?.getBoolean(EXTRA_STREAMING) ?: true
        val diarizationEnabled = recognizerIntent?.extras?.getBoolean(EXTRA_DIARIZATION) ?: false
        val noiseSuppressionEnabled =
            recognizerIntent?.extras?.getBoolean(EXTRA_NOISE_SUPPRESSION) ?: true
        val speechStartPadMs = 24000

        Log.i(
            TAG,
            "onStartListening: streamingEnabled=$streamingEnabled " +
                "isPartialResults=$isPartialResults diarizationEnabled=$diarizationEnabled " +
                "noiseSuppressionEnabled=$noiseSuppressionEnabled"
        )

        val streamingInFlight = AtomicBoolean(false)
        var lastStreamedSize = 0
        var lastStreamedTranscriptionIndex = -1
        var streamingAudioCtx = STREAMING_AUDIO_CTX_INITIAL

        // Stitching state: as the sliding window advances past audio that is no longer
        // visible to the streaming pass, we run a "lock-in" pass to transcribe that
        // orphaned chunk and append its text to confirmedPrefixText. The streaming
        // partial we emit is `confirmedPrefixText + " " + streamingTailText`.
        var confirmedPrefixText = ""
        var confirmedAudioEndSample = 0

        if (recordAndTranscribeJob?.isActive == true) {
            listener?.error(SpeechRecognizer.ERROR_RECOGNIZER_BUSY)
        }

        val attributionContext = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            this.createContext(
                ContextParams.Builder()
                    .setNextAttributionSource(listener?.callingAttributionSource)
                    .build()
            )
        } else {
            null
        }

        val sampleRate = 16000
        val encoding = AudioFormat.ENCODING_PCM_16BIT
        val bufferSize = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, encoding)

        val audioFormat = AudioFormat.Builder()
            .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
            .setSampleRate(sampleRate)
            .setEncoding(encoding)
            .build()

        val audioRecord = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && attributionContext != null) {
            try {
                AudioRecord.Builder()
                    .setContext(attributionContext)
                    .setAudioFormat(audioFormat)
                    .build()
            } catch (e: SecurityException) {
                throw SecurityException(e)
            }
        } else {
            try {
                AudioRecord.Builder()
                    .setAudioFormat(audioFormat)
                    .build()
            } catch (e: SecurityException) {
                throw SecurityException(e)
            }
        }

        // Per-session denoiser. RNNoise maintains internal noise-adaptation state, so a
        // fresh instance per recording is correct. If construction fails we fall back to
        // raw audio rather than failing the whole session — the user's transcription is
        // more important than a denoising pass.
        val denoiser: Denoiser? = if (noiseSuppressionEnabled) {
            try {
                Denoiser.create()
            } catch (e: Throwable) {
                Log.w(TAG, "failed to create RNNoise denoiser; falling back to raw audio", e)
                null
            }
        } else null


//        val model = when (recognizerIntent?.extras?.getString(RecognizerIntent.EXTRA_LANGUAGE_MODEL)) {
//            "TINY_EN_Q8_0" -> WhichModel.TINY_EN_Q8_0
//            "TINY_EN_Q4_0" -> WhichModel.TINY_EN_Q4_0
//            "BASE_Q8_0" -> WhichModel.BASE_Q8_0
//            "BASE_Q4_0" -> WhichModel.BASE_Q4_0
//
//            // These k-quants are currently very slow and inaccurate
//            "BASE_Q4K" -> WhichModel.BASE_Q4K
//            "BASE_Q2K" -> WhichModel.BASE_Q2K
//            else -> WhichModel.TINY_EN_Q4_0
//        }
//
//        val isMultilingual = when (model) {
//            WhichModel.TINY_EN_Q8_0 -> false
//            WhichModel.TINY_EN_Q4_0 -> false
//            WhichModel.BASE_Q8_0 -> true
//            WhichModel.BASE_Q4_0 -> true
//            WhichModel.BASE_Q4K -> true
//            WhichModel.BASE_Q2K -> true
//        }
//
//        println(model)
//
//        if (!isWhisperLoaded() || loadedModel != model) {
//            val modelResId = when (model) {
//                WhichModel.TINY_EN_Q8_0 -> R.raw.whisper_tiny_en_q8_0_model
//                WhichModel.TINY_EN_Q4_0 -> R.raw.whisper_tiny_en_q4_0_model
//                WhichModel.BASE_Q8_0 -> R.raw.whisper_base_q8_0_model
//                WhichModel.BASE_Q4_0 -> R.raw.whisper_base_q4_0_model
//                WhichModel.BASE_Q4K -> R.raw.whisper_base_q2k_model
//                WhichModel.BASE_Q2K -> R.raw.whisper_base_q2k_model
//            }
//            val configResId = when (model) {
//                WhichModel.TINY_EN_Q8_0 -> R.raw.whisper_tiny_en_config
//                WhichModel.TINY_EN_Q4_0 -> R.raw.whisper_tiny_en_config
//                WhichModel.BASE_Q8_0 -> R.raw.whisper_base_config
//                WhichModel.BASE_Q4_0 -> R.raw.whisper_base_config
//                WhichModel.BASE_Q4K -> R.raw.whisper_base_config
//                WhichModel.BASE_Q2K -> R.raw.whisper_base_config
//            }
//            val tokenizerResId = when (model) {
//                WhichModel.TINY_EN_Q8_0 -> R.raw.whisper_tiny_en_tokenizer
//                WhichModel.TINY_EN_Q4_0 -> R.raw.whisper_tiny_en_tokenizer
//                WhichModel.BASE_Q8_0 -> R.raw.whisper_base_tokenizer
//                WhichModel.BASE_Q4_0 -> R.raw.whisper_base_tokenizer
//                WhichModel.BASE_Q4K -> R.raw.whisper_base_tokenizer
//                WhichModel.BASE_Q2K -> R.raw.whisper_base_tokenizer
//            }
//
//            println(recognizerIntent?.extras?.getString(RecognizerIntent.EXTRA_LANGUAGE_MODEL))
//
//            val modelFileDescriptor = resources.openRawResourceFd(modelResId)
//            val configFileDescriptor = resources.openRawResourceFd(configResId)
//            val tokenizerFileDescriptor = resources.openRawResourceFd(tokenizerResId)
//
//            loadWhisper(
//                modelFileDescriptor.parcelFileDescriptor.detachFd(),
//                modelFileDescriptor.startOffset.toULong(),
//                modelFileDescriptor.length.toULong(),
//
//                configFileDescriptor.parcelFileDescriptor.detachFd(),
//                configFileDescriptor.startOffset.toULong(),
//                configFileDescriptor.length.toULong(),
//
//                tokenizerFileDescriptor.parcelFileDescriptor.detachFd(),
//                tokenizerFileDescriptor.startOffset.toULong(),
//                tokenizerFileDescriptor.length.toULong(),
//            )
//
//            modelFileDescriptor.close()
//            configFileDescriptor.close()
//            tokenizerFileDescriptor.close()
//
//            loadedModel = model
//
//            println("loaded model")
//        }


        var totalTranscriptionTime = 0L

        if (diarizationEnabled) {
            recordAndTranscribeJob = startDiarizationFlow(
                audioRecord = audioRecord,
                bufferSize = bufferSize,
                isPartialResults = isPartialResults == true,
                listener = listener,
                denoiser = denoiser,
            )
            return
        }

        recordAndTranscribeJob = recordAndTranscribeScope.launch recordAndTranscribe@{
            audioRecord.startRecording()
            isRecording.set(true)
            isSpeaking = false
            stopListening = false

            val audioRmsScope = CoroutineScope(Dispatchers.IO)
            transcribeJobs.clear()
            streamingJobs.clear()
            val transcriptions = mutableMapOf<Int, Transcription>()
            var transcriptionIndex by mutableIntStateOf(0)

            listener?.readyForSpeech(Bundle())

            while (isRecording.get() && isActive) {
                val rawBuffer = ShortArray(bufferSize)
                val numberOfShorts = audioRecord.read(rawBuffer, 0, bufferSize)
                // Apply RNNoise if enabled; otherwise pass the raw samples straight through.
                // The denoised chunk size may differ from numberOfShorts because RNNoise
                // operates on 480-sample 48 kHz frames and buffers partials; output lags
                // input by ~10 ms.
                val buffer: ShortArray = when {
                    numberOfShorts <= 0 -> ShortArray(0)
                    denoiser != null -> {
                        val raw = if (numberOfShorts == bufferSize) rawBuffer else rawBuffer.copyOf(numberOfShorts)
                        denoiser.process(raw)
                    }
                    numberOfShorts == bufferSize -> rawBuffer
                    else -> rawBuffer.copyOf(numberOfShorts)
                }

                // Snapshot transcriptionIndex once per loop iteration. The VAD coroutine
                // (audioRmsScope, on Dispatchers.IO) increments transcriptionIndex on its
                // "end" event, racing with this thread; without snapshotting we'd hit NPE
                // when the index advances between the null-check and the !! access below.
                val currentTranscriptionIndex = transcriptionIndex

                for (i in buffer.indices) {
                    if (transcriptions[currentTranscriptionIndex] == null) {
                        transcriptions[currentTranscriptionIndex] = Transcription(start = null, end = null, text = null)
                    }
                    transcriptions[currentTranscriptionIndex]!!.audioData.add(buffer[i])
                }

                if (!isActive) {
                    audioRecord.stop()
                    audioRecord.release()
                    denoiser?.close()
                    return@recordAndTranscribe
                }

                // Sliding-window streaming preview: while the current segment is still open
                // (VAD start fired, end has not), every ~1s of new audio we kick off a fast
                // greedy whisper pass on the segment-so-far and emit it as a streaming partial.
                if (streamingEnabled && isPartialResults == true) {
                    if (lastStreamedTranscriptionIndex != currentTranscriptionIndex) {
                        lastStreamedSize = 0
                        lastStreamedTranscriptionIndex = currentTranscriptionIndex
                        confirmedPrefixText = ""
                        confirmedAudioEndSample = 0
                    }
                    val streamingTranscription = transcriptions[currentTranscriptionIndex]
                    val streamingStart = streamingTranscription?.start
                    if (streamingTranscription != null
                        && streamingStart != null
                        && streamingTranscription.end == null
                        && !streamingInFlight.get()
                    ) {
                        val currentSize = streamingTranscription.audioData.size
                        if (currentSize - lastStreamedSize >= STREAMING_TRIGGER_SAMPLES) {
                            lastStreamedSize = currentSize
                            // Sliding window: at most STREAMING_WINDOW_SAMPLES of trailing audio.
                            val segmentSliceStart = (streamingStart.toInt() - speechStartPadMs).coerceAtLeast(0)
                            val windowSliceStart = (currentSize - STREAMING_WINDOW_SAMPLES).coerceAtLeast(0)
                            val sliceStart = maxOf(segmentSliceStart, windowSliceStart)
                            val sliceEnd = (currentSize - 1).coerceAtLeast(sliceStart)

                            // If the window has slid far enough past the last-confirmed point,
                            // grab the orphaned chunk so a lock-in pass can transcribe it.
                            val lockInStart = maxOf(segmentSliceStart, confirmedAudioEndSample)
                            val lockInEnd = sliceStart - 1
                            val lockInAudio = if (
                                lockInEnd >= lockInStart &&
                                (lockInEnd - lockInStart + 1) >= STREAMING_LOCKIN_GAP_SAMPLES
                            ) {
                                streamingTranscription.audioData
                                    .subList(lockInStart, lockInEnd + 1)
                                    .toShortArray()
                            } else null
                            val lockInUpToSample = if (lockInAudio != null) sliceStart else null

                            val snapshot = streamingTranscription.audioData
                                .subList(sliceStart, sliceEnd + 1)
                                .toShortArray()
                            val streamingForIndex = currentTranscriptionIndex
                            streamingInFlight.set(true)
                            val snapshotSize = snapshot.size
                            val ctxForThisPass = streamingAudioCtx
                            val launchedAtMs = currentTimeMillis()
                            Log.i(
                                TAG,
                                "streaming launch: idx=$streamingForIndex samples=$snapshotSize ctx=$ctxForThisPass"
                            )
                            val streamingJob = transcribeScope.launch streaming@{
                                try {
                                    // Lock-in pass: still on the cheap streaming path (greedy,
                                    // single-segment) so we never run the high-quality model
                                    // mid-stream. The segment-final pass that fires when VAD
                                    // detects end-of-speech is what re-transcribes the whole
                                    // segment with beam search and replaces the preview.
                                    if (lockInAudio != null && lockInUpToSample != null) {
                                        val lockText = whisperRepository.transcribePartialAudio(
                                            lockInAudio,
                                            ctxForThisPass,
                                        ).trim()
                                        if (lockText.isNotEmpty()) {
                                            confirmedPrefixText = if (confirmedPrefixText.isEmpty())
                                                lockText
                                            else
                                                "$confirmedPrefixText $lockText"
                                        }
                                        confirmedAudioEndSample = lockInUpToSample
                                        Log.i(
                                            TAG,
                                            "streaming lock-in: confirmedAudioEndSample=$confirmedAudioEndSample text=\"$lockText\""
                                        )
                                    }

                                    val tailText = whisperRepository.transcribePartialAudio(
                                        snapshot,
                                        ctxForThisPass,
                                    ).trim()
                                    val staleness = currentTimeMillis() - launchedAtMs

                                    // Adapt next pass's audio_ctx to keep staleness near the target.
                                    val previous = streamingAudioCtx
                                    streamingAudioCtx = when {
                                        staleness > STREAMING_TARGET_STALENESS_MS * 3 / 2 ->
                                            (streamingAudioCtx - STREAMING_AUDIO_CTX_STEP)
                                                .coerceAtLeast(STREAMING_AUDIO_CTX_MIN)
                                        staleness < STREAMING_TARGET_STALENESS_MS / 2 ->
                                            (streamingAudioCtx + STREAMING_AUDIO_CTX_STEP)
                                                .coerceAtMost(STREAMING_AUDIO_CTX_MAX)
                                        else -> streamingAudioCtx
                                    }
                                    if (streamingAudioCtx != previous) {
                                        Log.i(
                                            TAG,
                                            "streaming ctx adapt: $previous -> $streamingAudioCtx (staleness=${staleness}ms target=${STREAMING_TARGET_STALENESS_MS}ms)"
                                        )
                                    }

                                    if (!isActive || !isRecording.get()) {
                                        Log.i(TAG, "streaming dropped (cancelled) idx=$streamingForIndex staleness=${staleness}ms")
                                        return@streaming
                                    }
                                    val current = transcriptions[streamingForIndex]
                                    val ended = current?.end != null
                                    val combined = listOf(confirmedPrefixText, tailText)
                                        .filter { it.isNotEmpty() }
                                        .joinToString(" ")
                                    if (current != null && !ended && combined.isNotBlank()) {
                                        Log.i(TAG, "streaming emit idx=$streamingForIndex staleness=${staleness}ms ctx=$ctxForThisPass tail=\"$tailText\" prefix=\"$confirmedPrefixText\"")
                                        val bundle = Bundle().apply {
                                            putStringArrayList(
                                                SpeechRecognizer.RESULTS_RECOGNITION,
                                                arrayListOf(combined)
                                            )
                                            putBoolean(RESULTS_STREAMING_PARTIAL, true)
                                        }
                                        listener?.partialResults(bundle)
                                    } else {
                                        Log.i(
                                            TAG,
                                            "streaming dropped idx=$streamingForIndex staleness=${staleness}ms ended=$ended blank=${combined.isBlank()}"
                                        )
                                    }
                                } finally {
                                    streamingInFlight.set(false)
                                }
                            }
                            streamingJobs.add(streamingJob)
                        }
                    }
                }

                audioRmsScope.launch vadAndTranscribe@{
                    if (stopListening) {
                        isRecording.set(false)
                        listener?.endOfSpeech()

                        if ((transcriptionIndex != 0) && !autoStopRecognition && !isSpeaking) {
                            // break if we already transcribed once or more,
                            // is not auto stop recognition, and not speaking
                            // because we just want to stop then with the already transcribed parts.
                            // if there were no transcriptions then we assume VAD didn't work,
                            // so we transcribe everything all at once.
                            // if there were transcriptions, then VAD was working.
                            // in that case, we can break as if it was working at first it was
                            // probably working afterward, so processing the audio could
                            // be detrimental.
                            return@vadAndTranscribe
                        }

                        val transcription = transcriptions[transcriptionIndex]!!

                        if ((transcriptionIndex == 0) && !isSpeaking) {
                            transcription.start = 0.toDouble()
                        }

                        transcriptions[transcriptionIndex]!!.end =
                            (transcriptions[transcriptionIndex]!!.audioData.size - 1).toDouble()

                        val transcribeJob = transcribeScope.launch {
                            val timeBeforeTranscription = currentTimeMillis()

                            val transcriptionText =
                                whisperRepository.transcribeAudio(
                                    transcription.audioData.slice(
                                        ((transcription.start!!.toInt() - speechStartPadMs).coerceAtLeast(
                                            0
                                        ))..((transcription.end!!.toInt()).coerceAtMost(transcription.audioData.size - 1))
                                    )
                                        .toShortArray(),
                                )

                            transcription.text = transcriptionText

                            totalTranscriptionTime += currentTimeMillis() - timeBeforeTranscription

                            for (job in transcribeJobs.iterator().withIndex()) {
                                if (job.index < transcriptionIndex - 1) {
                                    job.value.join()
                                }
                            }

                            transcription.audioData.clear()

                            if (isPartialResults == true) {
                                val bundle = Bundle().apply {
                                    putStringArrayList(
                                        SpeechRecognizer.RESULTS_RECOGNITION,
                                        arrayListOf(
                                            transcription.text!!
                                        )
                                    )
                                }

                                if (!isActive) {
                                    return@launch
                                }

                                listener?.partialResults(bundle)
                            }
                        }

                        transcribeJobs.add(transcribeJob)

                        transcribeJob.start() // start transcribing this segment right now
                    } else {
                        sileroVadRepository.detect(buffer)?.forEach {
                            when (it.key) {
                                "start" -> {
                                    isSpeaking = true
                                    listener?.beginningOfSpeech()

                                    if (transcriptions[transcriptionIndex] == null) {
                                        transcriptions[transcriptionIndex] =
                                            Transcription(start = it.value, end = null, text = null)
                                    } else {
                                        transcriptions[transcriptionIndex]!!.start =
                                            it.value
                                    }
                                }

                                "end" -> {
                                    val transcription = transcriptions[transcriptionIndex]!!

                                    transcriptionIndex += 1
                                    sileroVadRepository.reset()

                                    if (transcription.end == null) {
                                        isSpeaking = false
                                        listener?.endOfSpeech()

                                        transcription.end = it.value

                                        val transcribeJob = transcribeScope.launch {
                                            val timeBeforeTranscription = currentTimeMillis()

                                            transcription.text =
                                                whisperRepository.transcribeAudio(
                                                    transcription.audioData.slice(
                                                        ((transcription.start!!.toInt() - speechStartPadMs).coerceAtLeast(
                                                            0
                                                        ))..((transcription.end!!.toInt()).coerceAtMost(transcription.audioData.size - 1))
                                                    )
                                                        .toShortArray(),
                                                )

                                            totalTranscriptionTime += currentTimeMillis() - timeBeforeTranscription

                                            for (job in transcribeJobs.iterator().withIndex()) {
                                                if (job.index < transcriptionIndex - 1) {
                                                    job.value.join()
                                                }
                                            }

                                            transcription.audioData.clear()

                                            if (isPartialResults == true) {
                                                val bundle = Bundle().apply {
                                                    putStringArrayList(
                                                        SpeechRecognizer.RESULTS_RECOGNITION,
                                                        arrayListOf(
                                                            transcription.text!!
                                                        )
                                                    )
                                                }

                                                if (!isActive) {
                                                    return@launch
                                                }

                                                listener?.partialResults(bundle)
                                            }
                                        }

                                        transcribeJobs.add(transcribeJob)

                                        transcribeJob.start() // start transcribing this segment right now

                                        if (autoStopRecognition) {
                                            isRecording.set(false)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (stopListening) {
                    break
                }
            }

            audioRecord.stop()
            audioRecord.release()
            denoiser?.close()

            sileroVadRepository.reset()

            if (!isActive) {
                return@recordAndTranscribe
            }

            runBlocking {
                transcribeJobs.forEach {
                    if (!isActive) {
                        return@forEach
                    }
                    it.join()
                }
            }

            if (!isActive) {
                return@recordAndTranscribe
            }

            val transcription = transcriptions.toSortedMap().values.joinToString { it.text ?: "" }

            val bundle = Bundle().apply {
                putStringArrayList(
                    SpeechRecognizer.RESULTS_RECOGNITION,
                    arrayListOf(
                        transcription
                    )
                )
            }


            try {
                listener?.results(bundle)
            } catch (e: RemoteException) {
                throw RuntimeException(e)
            }
        }
    }

    override fun onCancel(listener: Callback?) {
        recordAndTranscribeJob?.cancel()
        transcribeJobs.forEach {
            it.cancel()
        }
        streamingJobs.forEach {
            it.cancel()
        }
        sileroVadRepository.reset()
    }

    override fun onStopListening(listener: Callback?) {
        stopListening = true
    }

    private fun computeRms(samples: ShortArray): Float {
        if (samples.isEmpty()) return 0f
        var sumSq = 0.0
        for (s in samples) {
            val v = s.toDouble()
            sumSq += v * v
        }
        return kotlin.math.sqrt(sumSq / samples.size).toFloat()
    }

    override fun onDestroy() {
        super.onDestroy()
        recordAndTranscribeJob?.cancel()
        transcribeJobs.forEach {
            it.cancel()
        }
        streamingJobs.forEach {
            it.cancel()
        }
        runBlocking {
            whisperRepository.release()
            sileroVadRepository.release()
        }
        diarizationRepository.release()
    }

    /**
     * Diarization-mode audio loop. Replaces the VAD-driven recordAndTranscribe path
     * when [EXTRA_DIARIZATION] is true. Each AudioRecord buffer is fed straight into
     * a [DiarizationManager]; the manager emits per-speaker segment-closed events,
     * which we transcribe with the high-quality whisper context and surface as
     * partial bundles tagged with [RESULTS_SPEAKER_ID].
     */
    private fun startDiarizationFlow(
        audioRecord: AudioRecord,
        bufferSize: Int,
        isPartialResults: Boolean,
        listener: Callback?,
        denoiser: Denoiser?,
    ): Job = recordAndTranscribeScope.launch diarize@{
        isRecording.set(true)
        stopListening = false
        transcribeJobs.clear()
        streamingJobs.clear()

        listener?.readyForSpeech(Bundle())

        val segmenter: PyannoteSegmenterOnnxModel
        val embedder: PyannoteEmbedderOnnxModel
        try {
            segmenter = diarizationRepository.getSegmenter()
            embedder = diarizationRepository.getEmbedder()
            Log.i(TAG, "diarization models ready")
        } catch (e: ModelSchemaMismatchException) {
            Log.e(TAG, "diarization model schema mismatch", e)
            modelLoadError = "${e.modelName}: ${e.details}"
            withContext(Dispatchers.Main) {
                Toast.makeText(
                    this@MainRecognitionService,
                    "Transcribro: diarization disabled — ${e.modelName} model incompatible.",
                    Toast.LENGTH_LONG,
                ).show()
            }
            audioRecord.release()
            listener?.error(SpeechRecognizer.ERROR_CLIENT)
            return@diarize
        } catch (e: Throwable) {
            Log.e(TAG, "failed to load diarization models", e)
            audioRecord.release()
            listener?.error(SpeechRecognizer.ERROR_CLIENT)
            return@diarize
        }

        // Reset per-session state: each new diarization session starts speaker numbering
        // from 0. (The label store in the UI also resets via its own subscription.)
        speakerRegistry.reset()

        // Collect committed speaker segments so we can assemble a final transcript when
        // recognition ends. Append-only; the recordAndTranscribe coroutine is the sole
        // writer and onResults reads it once at the end.
        data class CommittedTurn(val speakerId: Int, val text: String)
        val committed = mutableListOf<CommittedTurn>()

        // The rolling 10-second audio window still contains audio that's already been
        // committed as a per-speaker segment. The streaming preview should not re-show
        // that audio, so we track the most-recent committed boundary and trim the
        // streaming snapshot to start from there.
        var lastCommitEndSample = 0L

        val manager = DiarizationManager(
            segmenter = segmenter,
            embedder = embedder,
            registry = speakerRegistry,
            onSegmentClosed = { event ->
                if (event.sampleRangeEnd > lastCommitEndSample) {
                    lastCommitEndSample = event.sampleRangeEnd
                }
                val job = transcribeScope.launch {
                    val text = whisperRepository.transcribeAudio(event.audio).trim()
                    if (text.isBlank()) return@launch
                    synchronized(committed) {
                        committed += CommittedTurn(event.globalSpeakerId, text)
                    }
                    if (!isPartialResults) return@launch
                    val bundle = Bundle().apply {
                        putStringArrayList(
                            SpeechRecognizer.RESULTS_RECOGNITION,
                            arrayListOf(text),
                        )
                        putInt(RESULTS_SPEAKER_ID, event.globalSpeakerId)
                    }
                    try {
                        listener?.partialResults(bundle)
                    } catch (e: RemoteException) {
                        Log.w(TAG, "partialResults RemoteException", e)
                    }
                }
                transcribeJobs.add(job)
            },
        )

        // Run a streaming preview alongside diarization so the user gets live feedback
        // during recording (same UX as the VAD flow). The streaming pass is unlabeled —
        // labeled commits land separately as the per-speaker segments close.
        val streamingInFlight = AtomicBoolean(false)
        var samplesSinceLastStreaming = 0L
        var streamingAudioCtx = STREAMING_AUDIO_CTX_INITIAL

        try {
            audioRecord.startRecording()
            while (isRecording.get() && isActive) {
                val rawBuffer = ShortArray(bufferSize)
                val n = audioRecord.read(rawBuffer, 0, bufferSize)
                if (n > 0) {
                    val rawChunk = if (n == bufferSize) rawBuffer else rawBuffer.copyOf(n)
                    val chunk = denoiser?.process(rawChunk) ?: rawChunk
                    if (chunk.isEmpty()) {
                        // Denoiser still buffering the first 10 ms of input — nothing to
                        // hand off this tick.
                        if (stopListening) {
                            isRecording.set(false)
                            listener?.endOfSpeech()
                            break
                        }
                        continue
                    }
                    manager.process(chunk)
                    samplesSinceLastStreaming += chunk.size

                    // Streaming preview every ~1s, on whatever audio is in the rolling
                    // window (which is also what DiarizationManager segments on).
                    if (isPartialResults &&
                        samplesSinceLastStreaming >= STREAMING_TRIGGER_SAMPLES &&
                        !streamingInFlight.get()
                    ) {
                        val rawSnapshot = manager.snapshotWindow()
                        val totalSamplesSeen = manager.samplesSeen()
                        val windowStartSample = totalSamplesSeen - rawSnapshot.size
                        // Skip audio that the user has already seen as a committed
                        // speaker segment. If the whole window is older than the last
                        // commit, there's nothing new to preview.
                        val skipFromStart = (lastCommitEndSample - windowStartSample)
                            .coerceAtLeast(0)
                            .coerceAtMost(rawSnapshot.size.toLong())
                            .toInt()
                        val snapshot = if (skipFromStart in 1 until rawSnapshot.size) {
                            rawSnapshot.copyOfRange(skipFromStart, rawSnapshot.size)
                        } else if (skipFromStart >= rawSnapshot.size) {
                            ShortArray(0)
                        } else {
                            rawSnapshot
                        }
                        // Suppress whisper on silent audio (avoids hallucinations like
                        // "you know what they…" / "♪♪" that surface after a speaker
                        // commits and the window's contents are mostly silence).
                        val rms = if (snapshot.size >= STREAMING_TRIGGER_SAMPLES) computeRms(snapshot) else 0f
                        if (snapshot.size >= STREAMING_TRIGGER_SAMPLES &&
                            rms < STREAMING_SILENCE_RMS_THRESHOLD
                        ) {
                            samplesSinceLastStreaming = 0L
                            if (isPartialResults) {
                                // Blank streaming partial clears any residual preview.
                                val bundle = Bundle().apply {
                                    putStringArrayList(
                                        SpeechRecognizer.RESULTS_RECOGNITION,
                                        arrayListOf(""),
                                    )
                                    putBoolean(RESULTS_STREAMING_PARTIAL, true)
                                }
                                try {
                                    listener?.partialResults(bundle)
                                } catch (e: RemoteException) {
                                    Log.w(TAG, "diarization silence-clear RemoteException", e)
                                }
                            }
                        } else if (snapshot.size >= STREAMING_TRIGGER_SAMPLES) {
                            samplesSinceLastStreaming = 0L
                            val ctxForThisPass = streamingAudioCtx
                            val launchedAtMs = currentTimeMillis()
                            streamingInFlight.set(true)
                            val streamingJob = transcribeScope.launch streaming@{
                                try {
                                    val text = whisperRepository.transcribePartialAudio(
                                        snapshot,
                                        ctxForThisPass,
                                    ).trim()
                                    val staleness = currentTimeMillis() - launchedAtMs
                                    val previous = streamingAudioCtx
                                    streamingAudioCtx = when {
                                        staleness > STREAMING_TARGET_STALENESS_MS * 3 / 2 ->
                                            (streamingAudioCtx - STREAMING_AUDIO_CTX_STEP)
                                                .coerceAtLeast(STREAMING_AUDIO_CTX_MIN)
                                        staleness < STREAMING_TARGET_STALENESS_MS / 2 ->
                                            (streamingAudioCtx + STREAMING_AUDIO_CTX_STEP)
                                                .coerceAtMost(STREAMING_AUDIO_CTX_MAX)
                                        else -> streamingAudioCtx
                                    }
                                    if (streamingAudioCtx != previous) {
                                        Log.i(
                                            TAG,
                                            "diarization streaming ctx adapt: $previous -> $streamingAudioCtx (staleness=${staleness}ms)"
                                        )
                                    }
                                    if (!isActive || !isRecording.get() || text.isBlank()) {
                                        return@streaming
                                    }
                                    Log.i(
                                        TAG,
                                        "diarization streaming emit staleness=${staleness}ms text=\"$text\""
                                    )
                                    val bundle = Bundle().apply {
                                        putStringArrayList(
                                            SpeechRecognizer.RESULTS_RECOGNITION,
                                            arrayListOf(text),
                                        )
                                        putBoolean(RESULTS_STREAMING_PARTIAL, true)
                                    }
                                    try {
                                        listener?.partialResults(bundle)
                                    } catch (e: RemoteException) {
                                        Log.w(TAG, "diarization streaming partial RemoteException", e)
                                    }
                                } finally {
                                    streamingInFlight.set(false)
                                }
                            }
                            streamingJobs.add(streamingJob)
                        }
                    }
                }
                if (stopListening) {
                    isRecording.set(false)
                    listener?.endOfSpeech()
                    break
                }
            }
        } finally {
            try { audioRecord.stop() } catch (_: Throwable) {}
            audioRecord.release()
            denoiser?.close()

            // Close any open per-speaker segments — emits one last transcribe job per.
            manager.flush()

            // Wait for queued transcription jobs to complete so the final results bundle
            // captures everything.
            runBlocking {
                transcribeJobs.toList().forEach {
                    if (it.isActive) it.join()
                }
            }

            // Assemble the final transcript: chronological list of "Speaker N: text" lines.
            // Same int-id payload the in-stream partials carried — the UI does the
            // display-name lookup on its side via SpeakerLabelStore.
            val final = synchronized(committed) {
                committed.joinToString(separator = "\n") { t ->
                    "Speaker ${t.speakerId + 1}: ${t.text}"
                }
            }
            val bundle = Bundle().apply {
                putStringArrayList(
                    SpeechRecognizer.RESULTS_RECOGNITION,
                    arrayListOf(final),
                )
            }
            try {
                listener?.results(bundle)
            } catch (e: RemoteException) {
                Log.w(TAG, "results RemoteException", e)
            }
        }
    }
}
