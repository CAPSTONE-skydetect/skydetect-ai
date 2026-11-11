import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("direct_setup", Path(__file__).with_name("direct_link_setup.py"))
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class DirectLinkTests(unittest.TestCase):
    def test_refuses_changed_baseline(self):
        with patch.object(setup, "value", side_effect=["802-3-ethernet", "manual"]):
            with self.assertRaisesRegex(RuntimeError, "baseline"):
                setup.apply(0, 0)

    def test_refuses_non_ethernet(self):
        with patch.object(setup, "value", return_value="802-11-wireless"):
            with self.assertRaisesRegex(RuntimeError, "Ethernet"):
                setup.apply(0, 0)

    def folder(self, path):
        path = Path(path)
        (path / "original.json").write_text(json.dumps({
            "ipv4.method": "auto", "ipv6.method": "auto", "rollback_unit": "test-rollback",
        }))
        return path

    def test_rollback_restores_only_changed_methods(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = self.folder(temporary)
            with patch.object(setup, "run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run:
                setup.restore(folder)
            self.assertEqual(run.call_args_list[0].args, (
                "nmcli", "connection", "modify", setup.UUID,
                "ipv4.method", "auto", "ipv6.method", "auto"))
            self.assertEqual(json.loads((folder / "result.json").read_text())["status"], "rolled_back")

    def test_failed_reachability_keeps_rollback(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = self.folder(temporary)
            def fake(*args, **kwargs):
                output = "100 (connected)" if "GENERAL.STATE" in args else setup.UUID
                return subprocess.CompletedProcess(args, 1 if args[0] == "ping" else 0, output, "")
            with patch.object(setup, "run", side_effect=fake) as run, patch.object(setup, "restore") as restore, patch.object(setup.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "warm-up"):
                    setup.verify(folder)
                restore.assert_called_once_with(folder)
                self.assertFalse(any(c.args[:2] == ("systemctl", "stop") for c in run.call_args_list))

    def test_stable_network_requires_25_samples_and_240_seconds(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = self.folder(temporary)
            def fake(*args, **kwargs):
                output = "100 (connected)" if "GENERAL.STATE" in args else setup.UUID
                return subprocess.CompletedProcess(args, 0, output, "")
            with patch.object(setup, "run", side_effect=fake) as run, patch.object(setup.time, "sleep") as sleep:
                setup.verify(folder)
            self.assertEqual(sum(c.args[0] for c in sleep.call_args_list), 240)
            self.assertEqual(sum(c.args[0] == "ping" for c in run.call_args_list), 26)
            self.assertEqual(run.call_args_list[-1].args, ("systemctl", "stop", "test-rollback.timer"))
            result = json.loads((folder / "result.json").read_text())
            self.assertEqual(result["status"], "network_verified")
            self.assertEqual(result["camera_validation"], "pending on Mac")

    def test_warmup_failure_does_not_shorten_stable_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = self.folder(temporary)
            replies = [(False, {"ping_exit": 1}), (True, {})] + [(True, {})] * 25
            with patch.object(setup, "run"), patch.object(setup, "probe", side_effect=replies), patch.object(setup.time, "sleep") as sleep:
                setup.verify(folder)
            self.assertEqual(sum(c.args[0] for c in sleep.call_args_list), 243)
            self.assertEqual(json.loads((folder / "result.json").read_text())["status"], "network_verified")

    def test_completed_old_timer_is_stopped_before_next_apply(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            folder = base / "direct-123"
            folder.mkdir()
            (folder / "original.json").write_text(json.dumps({"rollback_unit": "skydetect-direct-123-rollback"}))
            (folder / "result.json").write_text(json.dumps({"status": "rolled_back"}))
            with patch.object(setup, "BASE", base), patch.object(setup, "run", return_value=subprocess.CompletedProcess([], 3, "inactive", "")) as run:
                setup.retire_completed_rollbacks()
            self.assertEqual(run.call_args_list[-1].args, ("systemctl", "stop", "skydetect-direct-123-rollback.timer"))


if __name__ == "__main__":
    unittest.main()
