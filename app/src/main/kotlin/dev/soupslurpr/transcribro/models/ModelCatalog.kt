package dev.soupslurpr.transcribro.models

/**
 * What role a model plays in the pipeline. Each kind has its own loader contract — a
 * Whisper GGML model is loaded by [com.whispercpp.whisper.WhisperContext], the ONNX
 * models by their respective wrapper classes. The kind also determines which
 * per-feature picker(s) a given entry is eligible for.
 */
enum class ModelKind {
    /** Whisper.cpp transcription model (.bin / .gguf). */
    WhisperTranscriber,

    /** SileroVAD voice-activity detector (.onnx). Required for any recognition. */
    VoiceActivityDetector,

    /** pyannote-style frame-wise speaker segmenter (.onnx). Conversation-mode only. */
    SpeakerSegmenter,

    /** Speaker-embedding model — wespeaker, ECAPA, etc. (.onnx). Conversation-mode only. */
    SpeakerEmbedder,
}

/**
 * A downloadable model in the catalog.
 *
 * [url] must be an immutable reference (commit hash / release tag) — pulling against a
 * mutable ref like `main`/`master` defeats the SHA-256 pin. [sha256] is the lowercase
 * hex digest of the downloaded bytes; the installer fails fast on mismatch.
 *
 * [wrapperVersion] is a coarse compatibility tag bumped whenever a loader class's
 * ONNX I/O assumptions change. Models tagged with an older version are flagged as
 * incompatible at load time rather than crashing the audio loop.
 */
data class ModelEntry(
    val id: String,
    val kind: ModelKind,
    val displayName: String,
    val description: String?,
    val url: String,
    val sha256: String,
    val sizeBytes: Long,
    val wrapperVersion: Int = 1,
)

/**
 * Static registry of known model variants. Adding a new variant is a code change — both
 * the URL and SHA-256 need to be supplied. A future version may load this from a signed
 * remote manifest, but the trust model stays "we ship a list of pins; nothing else runs".
 */
object ModelCatalog {
    val entries: List<ModelEntry> = listOf(
        // --- Whisper transcribers -------------------------------------------------
        // All pinned to the same whisper.cpp HF commit so a future bump is one find-
        // and-replace. URLs are immutable refs and SHAs come from HF's x-linked-etag
        // header (the LFS digest), so a CDN-rewrite of the response is caught by the
        // installer's verification step.

        // tiny — fastest, OK for streaming preview, weakest quality.
        ModelEntry(
            id = "whisper-tiny.en-q5_1",
            kind = ModelKind.WhisperTranscriber,
            displayName = "Whisper tiny.en (q5_1)",
            description = "English-only, smallest + fastest. ~31 MB.",
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny.en-q5_1.bin",
            sha256 = "c77c5766f1cef09b6b7d47f21b546cbddd4157886b3b5d6d4f709e91e66c7c2b",
            sizeBytes = 32_166_155L,
        ),
        ModelEntry(
            id = "whisper-tiny.en-q8_0",
            kind = ModelKind.WhisperTranscriber,
            displayName = "Whisper tiny.en (q8_0)",
            description = "English-only, slightly better than q5_1. ~42 MB.",
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny.en-q8_0.bin",
            sha256 = "5bc2b3860aa151a4c6e7bb095e1fcce7cf12c7b020ca08dcec0c6d018bb7dd94",
            sizeBytes = 43_550_795L,
        ),
        ModelEntry(
            id = "whisper-tiny-q5_1",
            kind = ModelKind.WhisperTranscriber,
            displayName = "Whisper tiny (multilingual, q5_1)",
            description = "Multilingual, smallest. ~31 MB.",
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny-q5_1.bin",
            sha256 = "818710568da3ca15689e31a743197b520007872ff9576237bda97bd1b469c3d7",
            sizeBytes = 32_152_673L,
        ),
        ModelEntry(
            id = "whisper-tiny-q8_0",
            kind = ModelKind.WhisperTranscriber,
            displayName = "Whisper tiny (multilingual, q8_0)",
            description = "Multilingual, slightly better than q5_1. ~42 MB.",
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny-q8_0.bin",
            sha256 = "c2085835d3f50733e2ff6e4b41ae8a2b8d8110461e18821b09a15c40c42d1cca",
            sizeBytes = 43_537_433L,
        ),

        // base — better quality, still mobile-friendly.
        ModelEntry(
            id = "whisper-base.en-q5_1",
            kind = ModelKind.WhisperTranscriber,
            displayName = "Whisper base.en (q5_1)",
            description = "English-only, noticeably better than tiny. ~57 MB.",
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-base.en-q5_1.bin",
            sha256 = "4baf70dd0d7c4247ba2b81fafd9c01005ac77c2f9ef064e00dcf195d0e2fdd2f",
            sizeBytes = 59_721_011L,
        ),
        ModelEntry(
            id = "whisper-base.en-q8_0",
            kind = ModelKind.WhisperTranscriber,
            displayName = "Whisper base.en (q8_0)",
            description = "English-only, base size with higher quantization. ~78 MB.",
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-base.en-q8_0.bin",
            sha256 = "a4d4a0768075e13cfd7e19df3ae2dbc4a68d37d36a7dad45e8410c9a34f8c87e",
            sizeBytes = 81_781_811L,
        ),
        ModelEntry(
            id = "whisper-base-q5_1",
            kind = ModelKind.WhisperTranscriber,
            displayName = "Whisper base (multilingual, q5_1)",
            description = "Multilingual base model. ~57 MB.",
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-base-q5_1.bin",
            sha256 = "422f1ae452ade6f30a004d7e5c6a43195e4433bc370bf23fac9cc591f01a8898",
            sizeBytes = 59_707_625L,
        ),
        ModelEntry(
            id = "whisper-base-q8_0",
            kind = ModelKind.WhisperTranscriber,
            displayName = "Whisper base (multilingual, q8_0)",
            description = "Multilingual base model. ~78 MB.",
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-base-q8_0.bin",
            sha256 = "c577b9a86e7e048a0b7eada054f4dd79a56bbfa911fbdacf900ac5b567cbb7d9",
            sizeBytes = 81_768_585L,
        ),

        // small — best quality / speed tradeoff for most mid-range devices.
        ModelEntry(
            id = "whisper-small.en-q5_1",
            kind = ModelKind.WhisperTranscriber,
            displayName = "Whisper small.en (q5_1)",
            description = "English-only. Strong accuracy. ~181 MB.",
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-small.en-q5_1.bin",
            sha256 = "bfdff4894dcb76bbf647d56263ea2a96645423f1669176f4844a1bf8e478ad30",
            sizeBytes = 190_098_681L,
        ),
        ModelEntry(
            id = "whisper-small.en-q8_0",
            kind = ModelKind.WhisperTranscriber,
            displayName = "Whisper small.en (q8_0)",
            description = "English-only, higher quantization. ~252 MB.",
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-small.en-q8_0.bin",
            sha256 = "67a179f608ea6114bd3fdb9060e762b588a3fb3bd00c4387971be4d177958067",
            sizeBytes = 264_477_561L,
        ),
        ModelEntry(
            id = "whisper-small-q5_1",
            kind = ModelKind.WhisperTranscriber,
            displayName = "Whisper small (multilingual, q5_1)",
            description = "Multilingual. ~181 MB.",
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-small-q5_1.bin",
            sha256 = "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb",
            sizeBytes = 190_085_487L,
        ),
        ModelEntry(
            id = "whisper-small-q8_0",
            kind = ModelKind.WhisperTranscriber,
            displayName = "Whisper small (multilingual, q8_0)",
            description = "Multilingual, higher quantization. ~252 MB.",
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-small-q8_0.bin",
            sha256 = "49c8fb02b65e6049d5fa6c04f81f53b867b5ec9540406812c643f177317f779f",
            sizeBytes = 264_464_607L,
        ),

        // medium / large-turbo — quality-first; not all phones can hold them at runtime.
        ModelEntry(
            id = "whisper-medium.en-q5_0",
            kind = ModelKind.WhisperTranscriber,
            displayName = "Whisper medium.en (q5_0)",
            description = "English-only, high accuracy, heavy. ~514 MB.",
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-medium.en-q5_0.bin",
            sha256 = "76733e26ad8fe1c7a5bf7531a9d41917b2adc0f20f2e4f5531688a8c6cd88eb0",
            sizeBytes = 539_225_533L,
        ),
        ModelEntry(
            id = "whisper-medium-q5_0",
            kind = ModelKind.WhisperTranscriber,
            displayName = "Whisper medium (multilingual, q5_0)",
            description = "Multilingual, high accuracy, heavy. ~514 MB.",
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-medium-q5_0.bin",
            sha256 = "19fea4b380c3a618ec4723c3eef2eb785ffba0d0538cf43f8f235e7b3b34220f",
            sizeBytes = 539_212_467L,
        ),
        ModelEntry(
            id = "whisper-large-v3-turbo-q5_0",
            kind = ModelKind.WhisperTranscriber,
            displayName = "Whisper large-v3-turbo (q5_0)",
            description = "Multilingual, near-large-v3 quality at ~5x speed. ~548 MB.",
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3-turbo-q5_0.bin",
            sha256 = "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
            sizeBytes = 574_041_195L,
        ),
        ModelEntry(
            id = "whisper-large-v3-turbo-q8_0",
            kind = ModelKind.WhisperTranscriber,
            displayName = "Whisper large-v3-turbo (q8_0)",
            description = "Multilingual, top quality available. ~834 MB. Flagships only.",
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3-turbo-q8_0.bin",
            sha256 = "317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1",
            sizeBytes = 874_188_075L,
        ),

        // --- Voice activity ------------------------------------------------------
        ModelEntry(
            id = "silero-vad-v5",
            kind = ModelKind.VoiceActivityDetector,
            displayName = "Silero VAD v5",
            description = "Required for recognition. ~1.8 MB.",
            url = "https://github.com/snakers4/silero-vad/raw/bfdc0193023f121ea5b3cc7b176dbed570a68a59/src/silero_vad/data/silero_vad.onnx",
            sha256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
            sizeBytes = 1_807_575L,
        ),

        // --- Speaker segmenter ----------------------------------------------------
        ModelEntry(
            id = "pyannote-segmentation-3.0",
            kind = ModelKind.SpeakerSegmenter,
            displayName = "pyannote segmentation 3.0",
            description = "Frame-wise speaker activity on a 10s window. ~6 MB.",
            url = "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/9403a6902bb58e3d5ae8c7e77c3422de279db2e0/model.onnx",
            sha256 = "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
            sizeBytes = 5_905_412L,
        ),

        // --- Speaker embedder -----------------------------------------------------
        ModelEntry(
            id = "wespeaker-en-voxceleb-resnet34-LM",
            kind = ModelKind.SpeakerEmbedder,
            displayName = "wespeaker EN VoxCeleb ResNet34-LM",
            description = "256-d speaker embedding. English-trained. ~25 MB.",
            url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx",
            sha256 = "e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012",
            sizeBytes = 25_837_695L,
        ),
    )

    fun findById(id: String): ModelEntry? = entries.firstOrNull { it.id == id }

    fun forKind(kind: ModelKind): List<ModelEntry> = entries.filter { it.kind == kind }
}
