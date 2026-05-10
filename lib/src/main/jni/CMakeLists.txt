cmake_minimum_required(VERSION 3.10)

# Top-level orchestrator for the native libraries shipped in :lib.
# Each subdirectory builds one independent shared library; they share nothing.
project(transcribro_native)

add_subdirectory(whisper)
add_subdirectory(rnnoise)
