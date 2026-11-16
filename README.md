# Stenosis

Stenosis is a private speech recognition app for Android that runs on-device by
default. It is a fork of [Transcribro](https://github.com/soupslurpr/Transcribro) with
two focused modes:

- **Conversation transcription** — open the app, hit record, and get a live transcript of a
  conversation with optional speaker diarization (pyannote + wespeaker).
- **Keyboard transcription** — a voice-input IME for dictating into other apps, like the
  original Transcribro.

It uses whisper.cpp to run the OpenAI Whisper family of models, Silero VAD for voice
activity detection, RNNoise for optional noise suppression, and on-device pyannote
segmentation + wespeaker embeddings for speaker diarization. By default all processing
happens locally — no audio leaves the device.

## Server transcription (optional)

Each mode can instead send its transcription pass to a self-hosted, OpenAI-compatible
transcription server (Settings → Keyboard / Conversation → Server Transcription, with the
URL under Settings → Transcription Server). It is off by default; when it's on, recorded
speech segments are sent as WAV audio to the URL you configure, and only there.

- The request is a `multipart/form-data` POST to the URL exactly as entered (no path is
  appended) with `file` (16 kHz mono 16-bit WAV), `response_format=json`, `language=en`,
  and `model` only if one is set. An optional bearer token is sent as
  `Authorization: Bearer …`. Only the `text` field of the JSON reply is read.
- Only `https://` URLs are accepted.
- Voice activity detection, noise suppression and diarization still run on-device. The
  live streaming preview runs on-device too, and only if an on-device model is installed;
  the server is never polled for previews.
- If a request fails and an on-device transcription model is installed, that segment is
  transcribed locally instead; otherwise the session ends with an error.

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
