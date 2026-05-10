{
  description = "Transcribro — Android dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            android_sdk.accept_license = true;
          };
        };

        buildToolsVersion = "36.0.0";
        platformVersion = "36";
        ndkVersion = "27.2.12479018";
        cmakeVersion = "3.22.1";

        androidComposition = pkgs.androidenv.composeAndroidPackages {
          platformToolsVersion = "35.0.2";
          buildToolsVersions = [ buildToolsVersion ];
          platformVersions = [ platformVersion ];
          abiVersions = [ "x86_64" ];
          includeEmulator = true;
          includeNDK = true;
          ndkVersions = [ ndkVersion ];
          cmakeVersions = [ cmakeVersion ];
          includeSystemImages = true;
          systemImageTypes = [ "google_apis" ];
          includeSources = false;
        };

        androidSdk = androidComposition.androidsdk;
        androidSdkPath = "${androidSdk}/libexec/android-sdk";
        ndkPath = "${androidSdkPath}/ndk/${ndkVersion}";

        avdName = "transcribro_test";
        systemImage = "system-images;android-${platformVersion};google_apis;x86_64";
        # The debug build sets applicationIdSuffix = ".debug" in app/build.gradle.kts,
        # so the dev install lives alongside the prod app under this package id.
        appPackage = "dev.soupslurpr.transcribro.debug";

        envExports = ''
          export ANDROID_HOME="${androidSdkPath}"
          export ANDROID_SDK_ROOT="${androidSdkPath}"
          export ANDROID_NDK_HOME="${ndkPath}"
          export ANDROID_NDK_ROOT="${ndkPath}"
          export JAVA_HOME="${pkgs.jdk17.home}"
          export GRADLE_OPTS="-Dorg.gradle.project.android.aapt2FromMavenOverride=${androidSdkPath}/build-tools/${buildToolsVersion}/aapt2"
        '';

        ensureGradlew = ''
          if [ ! -x ./gradlew ]; then
            echo "error: ./gradlew not found in current directory ($PWD)" >&2
            echo "       run this from the project root" >&2
            exit 1
          fi
        '';

        ensureAvd = ''
          if ! avdmanager list avd 2>/dev/null | grep -q "Name: ${avdName}"; then
            echo "AVD '${avdName}' missing — creating it..."
            echo "no" | avdmanager create avd \
              -n "${avdName}" \
              -k "${systemImage}" \
              -d "pixel_5"
          fi
        '';

        mkApp = name: text: {
          type = "app";
          program = "${pkgs.writeShellApplication {
            inherit name;
            runtimeInputs = [ androidSdk pkgs.jdk17 ];
            text = envExports + text;
          }}/bin/${name}";
        };

        avdApp = mkApp "transcribro-avd" ''
          ${ensureAvd}
          echo "AVD '${avdName}' is ready"
        '';

        emulatorApp = mkApp "transcribro-emulator" ''
          ${ensureAvd}
          exec emulator -avd "${avdName}" -no-snapshot-save -gpu auto "$@"
        '';

        installApp = mkApp "transcribro-install" ''
          ${ensureGradlew}
          ./gradlew installDebug
        '';

        launchApp = mkApp "transcribro-launch" ''
          adb shell monkey -p "${appPackage}" -c android.intent.category.LAUNCHER 1 >/dev/null
          echo "launched ${appPackage}"
        '';

        runApp = mkApp "transcribro-run" ''
          ${ensureGradlew}
          ./gradlew installDebug
          adb shell monkey -p "${appPackage}" -c android.intent.category.LAUNCHER 1 >/dev/null
          echo "launched ${appPackage}"
        '';

        logcatApp = mkApp "transcribro-logcat" ''
          PID="$(adb shell pidof "${appPackage}" 2>/dev/null | tr -d '\r' || true)"
          if [ -n "$PID" ]; then
            echo "tailing logs for ${appPackage} (pid $PID); Ctrl+C to stop"
            exec adb logcat --pid="$PID" "$@"
          else
            echo "${appPackage} not running; tailing all logs filtered by tag"
            exec adb logcat "$@"
          fi
        '';
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            androidSdk
            jdk17
            kotlin
            gradle
          ];

          ANDROID_HOME = androidSdkPath;
          ANDROID_SDK_ROOT = androidSdkPath;
          ANDROID_NDK_HOME = ndkPath;
          ANDROID_NDK_ROOT = ndkPath;
          JAVA_HOME = "${pkgs.jdk17.home}";
          GRADLE_OPTS = "-Dorg.gradle.project.android.aapt2FromMavenOverride=${androidSdkPath}/build-tools/${buildToolsVersion}/aapt2";

          shellHook = ''
            echo "transcribro dev shell"
            echo "  android sdk: $ANDROID_HOME"
            echo "  android ndk: $ANDROID_NDK_HOME"
            echo "  java:        $(java -version 2>&1 | head -n1)"
            echo "  kotlin:      $(kotlin -version 2>&1)"
            echo "  gradle:      $(gradle --version | grep '^Gradle')"
            echo ""
            echo "available apps (run via 'nix run .#<name>'):"
            echo "  avd       — create the transcribro AVD if missing"
            echo "  emulator  — start the emulator (auto-creates AVD)"
            echo "  install   — build & install debug APK to running device"
            echo "  launch    — launch the installed debug app"
            echo "  run       — install + launch in one shot"
            echo "  logcat    — tail logs for ${appPackage}"
            echo ""
            echo "debug app id: ${appPackage} (installs alongside the prod app)"
          '';
        };

        apps = {
          avd = avdApp;
          emulator = emulatorApp;
          install = installApp;
          launch = launchApp;
          run = runApp;
          logcat = logcatApp;
          default = runApp;
        };
      });
}
