package dev.soupslurpr.transcribro.models

/**
 * Thrown by a model factory when the catalog entry it needs isn't on disk. The
 * recognition service catches this during preload and surfaces a "model not installed"
 * error rather than a generic load failure — the user fix is to install the model in
 * Settings, not to file a bug.
 */
class ModelNotInstalledException(val entry: ModelEntry) :
    RuntimeException("Model not installed: ${entry.id} (${entry.displayName})")
