#!/bin/bash
# Run on the Mac in a normal terminal; sudo asks for the Pi password there.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
host='doori@fe80::8dc1:9878:a2d:fa8a%en7'
options=(-6 -o BatchMode=yes -o StrictHostKeyChecking=yes -o HostKeyAlias=172.20.10.2 -o ConnectTimeout=6)
remote="$(ssh "${options[@]}" "$host" 'mktemp -d /tmp/skydetect-direct.XXXXXXXX')"
if [[ ! "$remote" =~ ^/tmp/skydetect-direct\.[A-Za-z0-9]+$ ]]; then
  echo 'Unexpected temporary directory; stopping.' >&2
  exit 1
fi
scp "${options[@]}" "$here/direct_link_setup.py" "doori@[fe80::8dc1:9878:a2d:fa8a%en7]:$remote/maintenance.py"
# Capture Pi uptime before sudo so time spent entering its password is compensated.
epoch="$(date +%s)"
ssh "${options[@]}" -t "$host" "start=\$(cut -d ' ' -f1 /proc/uptime); sudo /usr/bin/python3 '$remote/maintenance.py' apply '$epoch' \"\$start\""
