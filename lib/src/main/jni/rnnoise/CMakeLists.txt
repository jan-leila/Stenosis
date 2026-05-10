cmake_minimum_required(VERSION 3.10)
project(rnnoise C)

set(CMAKE_C_STANDARD 99)

# Vendored xiph/rnnoise submodule at repo root.
set(RNNOISE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../rnnoise)

# The default RNN weights aren't shipped in the xiph repo — they live in a separate
# `rnnoise_data-<sha>.tar.gz` tarball whose hash is pinned in `rnnoise/model_version`.
# Gradle's `downloadRnnoiseModel` task fetches + sha-verifies + extracts that tarball,
# then passes the extracted directory in via -DRNNOISE_DATA_DIR. Without it, the build
# can't link rnnoise_init_default(); fail fast with a clear message.
if(NOT DEFINED RNNOISE_DATA_DIR)
    message(FATAL_ERROR
        "RNNOISE_DATA_DIR not set. The Gradle externalNativeBuild config must pass "
        "-DRNNOISE_DATA_DIR=<path-to-extracted-rnnoise_data>. Run the "
        "`downloadRnnoiseModel` task or do a full `:lib:build`.")
endif()

set(RNNOISE_SOURCES
    ${RNNOISE_ROOT}/src/denoise.c
    ${RNNOISE_ROOT}/src/rnn.c
    ${RNNOISE_ROOT}/src/pitch.c
    ${RNNOISE_ROOT}/src/kiss_fft.c
    ${RNNOISE_ROOT}/src/celt_lpc.c
    ${RNNOISE_ROOT}/src/nnet.c
    ${RNNOISE_ROOT}/src/nnet_default.c
    ${RNNOISE_ROOT}/src/parse_lpcnet_weights.c
    ${RNNOISE_ROOT}/src/rnnoise_tables.c
    # Generated weights tables — extracted from the model tarball.
    ${RNNOISE_DATA_DIR}/src/rnnoise_data.c
    # Thin JNI surface (create / processFrame / destroy / frameSize).
    ${CMAKE_CURRENT_SOURCE_DIR}/jni.c
)

find_library(LOG_LIB log)

add_library(rnnoise SHARED ${RNNOISE_SOURCES})

target_compile_options(rnnoise PRIVATE -O3)
target_compile_definitions(rnnoise PRIVATE RNNOISE_BUILD)
target_include_directories(rnnoise PRIVATE
    ${RNNOISE_ROOT}/include
    ${RNNOISE_ROOT}/src
    ${RNNOISE_DATA_DIR}/src
)

# Mirror whisper's release hardening — symbol stripping, dead-code elimination, LTO.
if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    target_compile_options(rnnoise PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
    target_compile_options(rnnoise PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(rnnoise PRIVATE -Wl,--gc-sections)
    target_link_options(rnnoise PRIVATE -Wl,--exclude-libs,ALL)
    target_link_options(rnnoise PRIVATE -flto)
endif()

target_link_libraries(rnnoise ${LOG_LIB} android)
