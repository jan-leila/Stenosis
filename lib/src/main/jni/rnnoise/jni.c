// JNI bindings for xiph/rnnoise. Surface area is intentionally minimal — the
// resampling, frame buffering, and int16↔float conversion happen in Kotlin
// (see LibRnnoise.kt). This C layer just exposes the four functions we actually
// need from librnnoise.so.

#include <jni.h>
#include <stdint.h>
#include <android/log.h>
#include "rnnoise.h"

#define TAG "LibRnnoise"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

JNIEXPORT jlong JNICALL
Java_com_whispercpp_rnnoise_LibRnnoise_00024Companion_nativeCreate(
        JNIEnv *env, jobject thiz) {
    DenoiseState *st = rnnoise_create(NULL);
    if (!st) {
        LOGE("rnnoise_create returned NULL");
        return 0;
    }
    return (jlong)(uintptr_t)st;
}

JNIEXPORT void JNICALL
Java_com_whispercpp_rnnoise_LibRnnoise_00024Companion_nativeDestroy(
        JNIEnv *env, jobject thiz, jlong ptr) {
    if (ptr == 0) return;
    rnnoise_destroy((DenoiseState *)(uintptr_t)ptr);
}

// Processes exactly FRAME_SIZE (480) samples in-place. The float array is the
// 48 kHz frame in int16 amplitude range (NOT [-1, 1]) — this is the convention
// rnnoise was trained on, and the Kotlin wrapper preserves it across the
// resample. Returns the VAD probability (0..1); we ignore it on the Kotlin
// side for now but plumb it through in case a future caller wants it.
JNIEXPORT jfloat JNICALL
Java_com_whispercpp_rnnoise_LibRnnoise_00024Companion_nativeProcessFrame(
        JNIEnv *env, jobject thiz, jlong ptr, jfloatArray frame) {
    if (ptr == 0) return 0.0f;
    jsize len = (*env)->GetArrayLength(env, frame);
    if (len < 480) {
        LOGE("nativeProcessFrame got %d samples, need >=480", len);
        return 0.0f;
    }
    jfloat *buf = (*env)->GetFloatArrayElements(env, frame, NULL);
    if (!buf) return 0.0f;
    // rnnoise_process_frame allows in == out for in-place processing.
    float vad_prob = rnnoise_process_frame((DenoiseState *)(uintptr_t)ptr, buf, buf);
    (*env)->ReleaseFloatArrayElements(env, frame, buf, 0);
    return vad_prob;
}

JNIEXPORT jint JNICALL
Java_com_whispercpp_rnnoise_LibRnnoise_00024Companion_nativeFrameSize(
        JNIEnv *env, jobject thiz) {
    return rnnoise_get_frame_size();
}
