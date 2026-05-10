package dev.soupslurpr.transcribro.recognitionservice

import ai.onnxruntime.NodeInfo
import ai.onnxruntime.OrtSession

/**
 * Thrown when a loaded ONNX model's I/O signature doesn't match what the wrapper code
 * was written against. The point is to fail loudly and clearly when an upstream model
 * publishes a breaking change instead of running into a generic
 * `OrtException: Unexpected number of inputs, expected [1,3) found 4` deep inside the
 * audio loop.
 *
 * The model bundle is pinned by URL+SHA at build time (see DownloadModelsTask), so this
 * should normally only fire when a developer bumps the pin without updating the matching
 * wrapper. The user-facing contract is "show a toast, don't crash."
 */
class ModelSchemaMismatchException(
    val modelName: String,
    val details: String,
) : RuntimeException("Model '$modelName' schema mismatch: $details")

internal object OnnxSchemaValidator {
    /**
     * Verify that [session]'s input names match [expectedInputs] exactly (same set, same
     * names) and that it has [expectedOutputCount] outputs. Throws
     * [ModelSchemaMismatchException] on mismatch.
     */
    fun verify(
        modelName: String,
        session: OrtSession,
        expectedInputs: Set<String>,
        expectedOutputCount: Int,
    ) {
        val actualInputs = session.inputInfo.keys.toSet()
        if (actualInputs != expectedInputs) {
            throw ModelSchemaMismatchException(
                modelName,
                "expected inputs=$expectedInputs, got $actualInputs",
            )
        }
        val actualOutputs = session.outputInfo.size
        if (actualOutputs != expectedOutputCount) {
            throw ModelSchemaMismatchException(
                modelName,
                "expected $expectedOutputCount outputs, got $actualOutputs " +
                    "(${session.outputInfo.keys})",
            )
        }
    }

    /** Render a NodeInfo for diagnostics (used in error messages). */
    fun describe(info: NodeInfo): String = "${info.name}: ${info.info}"
}
