"""클라우드 MediaMTX 의 pi 경로에 영상이 들어오는 동안 상태 파일(state.json)을 갱신한다.

clip-bridge 는 이 파일의 last_media_progress_at 이 20초 안이어야 클립을 만든다.
로컬에서는 hardware/raspberry-pi/control.py (Mac) 또는 .skydetect-pi/heartbeat.ps1 (Windows) 이 하던 일이다.
"""
import json
import os
import time
import urllib.request
from pathlib import Path

API = os.environ.get("MTX_API", "http://mediamtx:9997/v3/paths/list")
STATE = Path(os.environ.get("PI_RELAY_STATE", "/runtime/status/state.json"))
HLS_URL = os.environ.get("PI_HLS_URL", "/live/pi/index.m3u8")

started = int(time.time())
last = -1
STATE.parent.mkdir(parents=True, exist_ok=True)
while True:
    try:
        items = json.load(urllib.request.urlopen(API, timeout=3)).get("items", [])
        path = next((p for p in items if p.get("name") == "pi"), None)
        received = path.get("inboundBytes", 0) if path else 0
        # MediaMTX 가 재시작되면 수신 바이트가 0 부터 다시 센다. "이전보다 크면"으로 비교하면
        # 재시작 뒤 상태가 갱신되지 않아 clip-bridge 가 클립 요청을 모두 503 으로 거절한다.
        if path and path.get("ready") and received != last:
            last = received
            state = dict(instance_id="cloud-heartbeat", status="running", started_at_epoch=started,
                         hls_url=HLS_URL, last_media_progress_at=time.time(), capture_children=[])
            tmp = STATE.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state), encoding="utf-8")
            tmp.replace(STATE)
    except Exception as exc:  # 중계 서버 재시작 중에는 다음 주기에 다시 본다
        print(f"heartbeat: {exc}", flush=True)
    time.sleep(1)
