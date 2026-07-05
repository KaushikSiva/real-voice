#!/usr/bin/env bash
set -euo pipefail

ffmpeg -hide_banner -f avfoundation -list_devices true -i "" 2>&1 || true

