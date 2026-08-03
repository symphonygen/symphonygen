#!/bin/bash
# Set up headless MuseScore for MIDI-to-audio rendering (used by utils/midi2audio.py).
# Expects the MuseScore AppImage to be placed in $ASSET_DIR beforehand.
set -e

ASSET_DIR="${ASSET_DIR:-asset}"

# 1. Extract the AppImage and make it executable
cd "$ASSET_DIR"
./MuseScore-3.6.2.548021370-x86_64.AppImage --appimage-extract
chmod -R +x squashfs-root/

# 2. Install dependencies
# libmp3lame0 is required for MP3 export — without it MuseScore pops a
# "locate libmp3lame.so.0?" dialog and hangs forever under Xvfb.
DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y \
    xvfb \
    libmp3lame0 \
    libnss3 \
    libfontconfig1 \
    libasound2 \
    libegl1-mesa \
    libgl1-mesa-glx \
    libgles2-mesa \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libgtk-3-0

# 3. utils/midi2audio.py will now detect and run MuseScore in headless mode:
# xvfb-run --auto-servernum $ASSET_DIR/squashfs-root/AppRun -o {output} {input}
