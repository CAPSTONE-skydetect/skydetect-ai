"""Disposable MySQL/Spring/real FFmpeg check; no real footage or live DB writes."""
from datetime import datetime, timezone
import hashlib
import http.cookiejar
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

HERE = Path(__file__).resolve().parent
AI_REPO = HERE.parents[1]
ENV = AI_REPO / 'docker/.env'
BRIDGE = HERE / 'bridge'
PROJECT = 'skydetect-spring-check-' + uuid.uuid4().hex[:8]
REPORT = AI_REPO / 'reports/pi-integration/spring_clip_check.json'

def client_at(base):
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    def call(path, data=None, headers=None):
        try:
            with opener.open(urllib.request.Request(base+path, data=data, headers=headers or {}), timeout=10) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()
    def token():
        return next(urllib.parse.unquote(cookie.value) for cookie in jar if cookie.name == 'XSRF-TOKEN')
    return call, token

def login(call, token, settings):
    assert call('/api/auth/me')[0] == 401
    form = urllib.parse.urlencode({'username':settings['OPERATOR_USERNAME'],
                                  'password':settings['OPERATOR_PASSWORD']}).encode()
    assert call('/api/auth/login', form, {'Content-Type':'application/x-www-form-urlencoded',
                                          'X-XSRF-TOKEN':token()})[0] == 200

def wait_for_backend(base):
    for _ in range(50):
        try:
            if client_at(base)[0]('/api/auth/me')[0] == 401:
                return
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError('Isolated Spring backend did not become ready.')

def main():
    browser_mode = '--browser' in sys.argv
    settings = dict(line.split('=',1) for line in ENV.read_text().splitlines() if '=' in line)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if browser_mode:
        settings.update(OPERATOR_USERNAME='browser-check', OPERATOR_PASSWORD=uuid.uuid4().hex)
    with tempfile.TemporaryDirectory(prefix='spring-clips-', dir=REPORT.parent) as folder:
        runtime = Path(folder)
        for sub in ('recordings/pi', 'status', 'clips'):
            (runtime/sub).mkdir(parents=True)
        spec = {
            'services': {
                'mysql': {'image':'mysql:8.4', 'environment':{'MYSQL_DATABASE':'skydetect',
                    'MYSQL_USER':'${DB_USERNAME}', 'MYSQL_PASSWORD':'${DB_PASSWORD}',
                    'MYSQL_ROOT_PASSWORD':'${DB_ROOT_PASSWORD}'}, 'volumes':['check-db:/var/lib/mysql'],
                    'healthcheck':{'test':['CMD-SHELL','mysqladmin ping -h 127.0.0.1 --silent'],
                                   'interval':'2s','timeout':'3s','retries':35}},
                'backend': {'image':'skydetect-pi-local-backend', 'ports':['127.0.0.1::8080'],
                    'environment':{'SPRING_DATASOURCE_URL':'jdbc:mysql://mysql:3306/skydetect',
                        'SPRING_DATASOURCE_USERNAME':'${DB_USERNAME}', 'SPRING_DATASOURCE_PASSWORD':'${DB_PASSWORD}',
                        'SKYDETECT_AUTH_INITIAL_OPERATOR_USERNAME':settings['OPERATOR_USERNAME'] if browser_mode else '${OPERATOR_USERNAME}',
                        'SKYDETECT_AUTH_INITIAL_OPERATOR_PASSWORD':settings['OPERATOR_PASSWORD'] if browser_mode else '${OPERATOR_PASSWORD}',
                        'SKYDETECT_CLIPS_WORKER_URL':'http://clip-bridge:8080',
                        'SKYDETECT_CLIPS_DIRECTORY':'/pi-clips'},
                    'volumes':[str(runtime/'clips')+':/pi-clips:ro'],
                    'depends_on':{'mysql':{'condition':'service_healthy'}}},
                'clip-bridge': {'image':'skydetect-pi-local-ai-rf', 'command':['python','/bridge/server.py'],
                    'volumes':[str(BRIDGE)+':/bridge:ro',str(runtime)+':/runtime'],
                    'user':f'{os.getuid()}:{os.getgid()}'},
            },
            'volumes':{'check-db':{}},
        }
        if browser_mode:
            for model in ('rf', 'minirocket'):
                spec['services']['ai-'+model] = {'image':'skydetect-pi-local-ai-'+model,
                    'environment':{'SKYDETECT_UPLOAD_DIR':'/app/storage/uploads',
                                   'SKYDETECT_TRACK_OUTPUT_DIR':'/app/artifacts/manual_tracks'}}
            spec['services']['frontend'] = {'image':'skydetect-pi-local-frontend',
                'ports':['127.0.0.1::80'], 'depends_on':['backend','clip-bridge','ai-rf','ai-minirocket'],
                'volumes':[str(BRIDGE.parent/'nginx.pi.conf')+':/etc/nginx/conf.d/default.conf:ro']}
        config = runtime/'compose.json'
        config.write_text(json.dumps(spec))
        compose = ['docker','compose','-p',PROJECT,'--env-file',str(ENV),'-f',str(config)]
        try:
            subprocess.run(compose+['up','-d','--no-build'], check=True, timeout=100)
            base = 'http://' + subprocess.check_output(compose+['port','backend','8080'], text=True).strip()
            wait_for_backend(base)
            call, token = client_at(base)
            assert call('/api/clips')[0] == 401
            login(call, token, settings)
            assert call('/api/clips', b'{}', {'Content-Type':'application/json'})[0] == 403
            assert call('/api/clips/missing')[0] == 404
            headers = {'Content-Type':'application/json', 'X-XSRF-TOKEN':token()}
            status, body = call('/api/clips', b'{}', headers)
            assert status == 202
            failed_id = json.loads(body)['clipId']
            for _ in range(20):
                failed = json.loads(call('/api/clips/'+failed_id)[1])
                if failed['status'] != 'PENDING':
                    break
                time.sleep(0.25)
            assert failed['status'] == 'FAILED', failed
            source = runtime/'source.mp4'
            generator = ['ffmpeg','-v','error','-f','lavfi','-i','color=c=0x263b50:s=1920x1080:r=30:d=12']
            if browser_mode:
                generator += ['-f','lavfi','-i','testsrc2=s=120x96:r=30:d=12',
                              '-filter_complex', "overlay=x='450+60*t':y='440+40*sin(t)'"]
            subprocess.run(generator + ['-c:v','libx264','-preset','ultrafast','-g','30','-bf','0',
                            '-movflags','+empty_moov+frag_keyframe+default_base_moof',str(source)],
                           check=True, timeout=45)
            now = time.time()
            for index in range(3):
                stamp = datetime.fromtimestamp(now-9+12*index, timezone.utc).strftime('%Y-%m-%d_%H-%M-%S-%f_%z')
                segment = runtime/'recordings/pi'/f'{stamp}.mp4'
                shutil.copyfile(source, segment)
                os.utime(segment, (now-5, now-5))
            (runtime/'status/state.json').write_text(json.dumps({'status':'running','last_media_progress_at':time.time()}))
            status, body = call('/api/clips', b'{"displayedMediaTimeS":2}', headers)
            assert status == 202, (status, body)
            accepted = json.loads(body)
            assert accepted['status'] == 'PENDING'
            route = '/api/clips/'+accepted['clipId']
            for _ in range(40):
                status, body = call(route)
                assert status == 200
                ready = json.loads(body)
                if ready['status'] != 'PENDING':
                    break
                time.sleep(1)
            assert ready['status'] == 'READY', ready
            assert ready['timing']['requestedSpanSeconds'] == 13
            assert len(ready['meta']['framePts']) == ready['meta']['frameCount']
            status, data = call(ready['videoUrl'], headers={'Range':'bytes=0-1023'})
            assert status == 206 and len(data) == 1024
            assert client_at(base)[0](ready['videoUrl'])[0] == 401
            assert any(row['clipId'] == ready['clipId'] for row in json.loads(call('/api/clips')[1])['clips'])
            import sqlite3
            with sqlite3.connect(runtime/'local-bridge/jobs.sqlite3') as sqlite:
                assert sqlite.execute('SELECT COUNT(*) FROM clips').fetchone()[0] == 0
            subprocess.run(compose+['restart','backend'],check=True,timeout=30)
            base = 'http://' + subprocess.check_output(compose+['port','backend','8080'], text=True).strip()
            wait_for_backend(base)
            call, token = client_at(base)
            login(call, token, settings)
            assert json.loads(call(route)[1]) == ready
            report = {'scope':'synthetic finalized MediaMTX-format segments, isolated MySQL project',
                      'spring_status':'READY', 'offline_failure':'FAILED', 'frame_count':ready['meta']['frameCount'],
                      'duration_ms':ready['meta']['durationMs'], 'range_status':206,
                      'video_sha256':hashlib.sha256((runtime/'clips'/(ready['clipId']+'.mp4')).read_bytes()).hexdigest(),
                      'restart_preserved':True, 'legacy_sqlite_clip_rows':0, 'pi_used':False}
            report_path = REPORT.with_name('spring_browser_fixture.json') if browser_mode else REPORT
            report_path.write_text(json.dumps(report,indent=2))
            print(json.dumps(report,indent=2))
            if browser_mode:
                port = subprocess.check_output(compose+['port','frontend','80'],text=True).strip().rsplit(':',1)[1]
                print(f'BROWSER_CHECK http://localhost:{port}/', flush=True)
                print(f'Temporary test account: {settings["OPERATOR_USERNAME"]} / {settings["OPERATOR_PASSWORD"]}', flush=True)
                print(f'Clip: {ready["clipId"]}', flush=True)
                input('Press Enter to remove only this disposable test stack: ')
        except Exception:
            subprocess.run(compose+['logs','--tail','35','backend','clip-bridge'],timeout=15)
            raise
        finally:
            # Only the randomly named disposable test project is removed.
            subprocess.run(compose+['down','-v'],check=True,timeout=90)

if __name__ == '__main__':
    main()
