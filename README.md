# Stenosis

Stenosis is a private and on-device speech recognition app for Android. It is a fork of
[Transcribro](https://github.com/soupslurpr/Transcribro) with two focused modes:

- **Conversation transcription** — open the app, hit record, and get a live transcript of a
  conversation with optional speaker diarization (pyannote + wespeaker).
- **Keyboard transcription** — a voice-input IME for dictating into other apps, like the
  original Transcribro.

It uses whisper.cpp to run the OpenAI Whisper family of models, Silero VAD for voice
activity detection, RNNoise for optional noise suppression, and on-device pyannote
segmentation + wespeaker embeddings for speaker diarization. All processing happens
locally — no audio leaves the device.

## Language support

Stenosis currently ships with the `tiny.en` Whisper model — English only. Multilingual
support is possible but not yet wired up.

## Build

This project uses a nix flake for the dev shell. From the repo root:

```
nix develop
./gradlew :app:assembleDebug
```

Or use the bundled flake apps:

```
nix run .#install   # build + install the debug APK to a connected device
nix run .#run       # install + launch
nix run .#logcat    # tail logcat for the debug app id
```

The debug build uses `applicationIdSuffix = ".debug"` so it installs alongside any
prod build.

## Attribution

Stenosis is a fork of [soupslurpr/Transcribro](https://github.com/soupslurpr/Transcribro)
(ISC License). The original keyboard transcription path, JNI whisper.cpp bindings, and
much of the recognition service plumbing come from there.
