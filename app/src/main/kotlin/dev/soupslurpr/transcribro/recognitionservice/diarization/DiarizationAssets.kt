package dev.soupslurpr.transcribro.recognitionservice.diarization

/**
 * Asset paths for the diarization ONNX models.
 *
 * The model files are NOT committed to the repo. They're downloaded into a generated
 * assets dir by the `downloadDiarizationModels` Gradle task (see app/build.gradle.kts),
 * which AGP includes as an additional asset source for every variant. So a normal
 * `./gradlew assembleDebug` will fetch them on first build and cache them after.
 *
 * The loaders ([PyannoteSegmenterOnnxModel] / [PyannoteEmbedderOnnxModel]) assume the
 * sherpa-onnx I/O names ("x"/"y" and "feats"/"embs"). If the download URLs in
 * `downloadDiarizationModels` are swapped to a different export, the loaders' input
 * names need to match.
 */
object DiarizationAssets {
    /**
     * pyannote/segmentation-3.0 — frame-wise speaker activity detection on a 10s window.
     *
     * Download:
     *   https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx
     *
     * Or the int8-quantized variant (smaller, slightly less accurate):
     *   https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.int8.onnx
     */
    const val SEGMENTATION_ASSET: String = "models/pyannote/segmentation-3.0.onnx"

    /**
     * wespeaker_en_voxceleb_resnet34_LM — 256-dim speaker embedding from kaldi fbank.
     *
     * Download (sherpa-onnx release):
     *   https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx
     */
    const val EMBEDDER_ASSET: String = "models/pyannote/wespeaker-en-voxceleb-resnet34-LM.onnx"
}
