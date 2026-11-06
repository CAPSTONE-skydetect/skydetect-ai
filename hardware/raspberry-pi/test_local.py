import importlib.util
import io
from pathlib import Path
import tempfile
import shutil
import subprocess
import unittest
from unittest.mock import patch, Mock

HERE = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

relay = load('relay_test', HERE / 'control.py')
bridge = load('bridge_test', HERE / 'bridge/server.py')

class LocalTests(unittest.TestCase):
    def test_playlist_query_string(self):
        responses = [b'#EXTM3U\nvideo1_stream.m3u8?session=abc\n',
                     b'#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:3\n#EXT-X-PART:DURATION=0.2,URI="part9.mp4"\n']
        with patch.object(relay.urllib.request, 'urlopen', side_effect=[io.BytesIO(x) for x in responses]) as get:
            self.assertEqual(relay._fetch_playlist('http://127.0.0.1/index.m3u8'),
                             ('#EXT-X-MEDIA-SEQUENCE:3', 'part9.mp4'))
            self.assertIn('?session=abc', get.call_args.args[0])

    def test_long_capture_not_per_minute(self):
        self.assertNotIn('"60000"', relay.REMOTE_CAMERA)
        compile(relay.REMOTE_CAMERA, '<remote>', 'exec')

    def test_reader_session_not_media_progress(self):
        def fingerprint(session):
            content = ('#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:3\n'
                       '#EXT-X-PART:DURATION=0.2,URI="part9.mp4?session=' + session + '"\n').encode()
            with patch.object(relay.urllib.request, 'urlopen', return_value=io.BytesIO(content)):
                return relay._fetch_playlist('http://127.0.0.1/stream.m3u8')
        self.assertEqual(fingerprint('first'), fingerprint('second'))

    def test_segment_names_reject_paths(self):
        self.assertIsNone(bridge.parse_segment_time(Path('../../secret.mp4')))
        self.assertIsNotNone(bridge.parse_segment_time(Path('2026-11-06_16-00-00-000000_+0900.mp4')))

    def test_csrf(self):
        self.assertFalse(bridge.csrf_ok('', None))
        self.assertFalse(bridge.csrf_ok('XSRF-TOKEN=abc', 'other'))
        self.assertTrue(bridge.csrf_ok('XSRF-TOKEN=abc', 'abc'))

    def test_quiet_active_segment_is_not_finalized(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'2026-11-06_16-00-00-000000_+0900.mp4'
            source.write_bytes(b'first-flushed-part')
            target = root/'copy/segment.mp4'
            with patch.object(bridge, 'stable_file', return_value=True):
                self.assertFalse(bridge._copy_stable_segment(source, target))
                self.assertFalse(target.exists())
                # A later flush completes the file before MediaMTX opens its successor.
                source.write_bytes(b'complete-video-with-all-parts')
                (root/'2026-11-06_16-00-11-000000_+0900.mp4').write_bytes(b'new-part')
                self.assertTrue(bridge._copy_stable_segment(source, target))
                self.assertEqual(target.read_bytes(), source.read_bytes())

    def test_irrelevant_segments_are_excluded(self):
        paths = [Path('2026-11-06_16-00-00-000000_+0900.mp4'),
                 Path('2026-11-06_16-00-11-000000_+0900.mp4'),
                 Path('2026-11-06_16-00-22-000000_+0900.mp4')]
        start = bridge.parse_segment_time(paths[0])
        with patch.object(bridge, 'probe_video', return_value={'format':{'duration':11}}):
            self.assertEqual(bridge.parts_overlapping_window(paths, start+12, start+20), [paths[1]])

    def test_real_pts_gap_is_still_rejected(self):
        part = Path('2026-11-06_16-00-00-000000_+0900.mp4')
        media = {'width':1920, 'height':1080, 'start_time':'0', 'duration':'0.3'}
        frames = [{'pts':value,'duration':1/30,'key':int(value == 0)} for value in [0,1/30,0.233]]
        with patch.object(bridge, '_ffprobe_frames', return_value=(media, frames)):
            with self.assertRaisesRegex(RuntimeError, 'intra_segment_pts_gap'):
                bridge.inspect_segments([part])

    def test_invalid_roi_never_starts_analysis(self):
        row = {'status':'READY', 'metadata_json':'{"frameCount":390,"width":1920,"height":1080,"durationMs":12967}',
               'frame_map_json':'[0.033]'}
        handler = Mock()
        with patch.object(bridge, '_db_clip', return_value=row), patch.object(bridge.threading, 'Thread') as worker:
            bridge.Handler._create_analysis(handler, 'clip_'+'a'*32,
                {'model':'rf', 'targetBbox':[1900,10,48,48], 'initFrameIndex':0, 'maxSeconds':3})
            handler._error.assert_called_once()
            self.assertEqual(handler._error.call_args.args[0], 400)
            worker.assert_not_called()

    def test_invalid_interval_never_starts_analysis(self):
        row = {'status':'READY', 'metadata_json':'{"frameCount":390,"width":1920,"height":1080,"durationMs":12967}',
               'frame_map_json':'[0.033]'}
        handler = Mock()
        with patch.object(bridge, '_db_clip', return_value=row), patch.object(bridge.threading, 'Thread') as worker:
            bridge.Handler._create_analysis(handler, 'clip_'+'a'*32,
                {'model':'minirocket', 'targetBbox':[50,50,48,48], 'initFrameIndex':0, 'maxSeconds':30})
            self.assertEqual(handler._error.call_args.args[0], 400)
            worker.assert_not_called()

    def test_restart_marks_incomplete_jobs_failed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.multiple(bridge, DB_PATH=root/'db.sqlite', CLIPS_DIR=root/'clips',
                                STAGE_DIR=root/'stage', PIN_DIR=root/'pins'):
                bridge.initialize_db()
                with bridge.connect() as db:
                    self.assertEqual(db.execute('PRAGMA journal_mode').fetchone()[0], 'delete')
                    db.execute("INSERT INTO clips(clip_id,status,created_at,ready_at,trigger_server_epoch,display_mapping,requested_start_epoch,requested_end_epoch,timing_json) VALUES('clip_test','PENDING',0,1,0,'unknown',0,1,'{}')")
                bridge.initialize_db()
                with bridge.connect() as db:
                    row = db.execute('SELECT status,error FROM clips').fetchone()
                self.assertEqual(row['status'], 'FAILED')
                self.assertIn('restarted', row['error'])
                with self.assertRaises(bridge.sqlite3.ProgrammingError):
                    db.execute('SELECT 1')

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg required')
    def test_real_encoder_non_frame_aligned_cut(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            part = root / '2026-11-06_16-00-00-000000_+0900.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                            'color=c=black:s=1920x1080:r=30:d=4', '-c:v', 'libx264',
                            '-preset', 'ultrafast', str(part)], check=True, timeout=30)
            start = bridge.parse_segment_time(part) + 1.123
            result = bridge._extract_clip([part], root, root/'result.mp4', start, start+2)
            self.assertEqual(result['metadata']['frameCount'], 60)
            self.assertEqual(len(result['frame_map']), 60)

if __name__ == '__main__':
    unittest.main()
