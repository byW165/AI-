import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from scripts import launch_dashboard as launcher


class LauncherTests(unittest.TestCase):
    def test_busy_port_is_skipped_without_stopping_owner(self):
        with socket.socket() as busy, socket.socket() as free:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            free.bind(("127.0.0.1", 0))
            first, second = busy.getsockname()[1], free.getsockname()[1]
            free.close()
            self.assertEqual(launcher.available_port([first, second]), second)
            self.assertEqual(busy.getsockname()[1], first)

    def test_all_busy_ports_fail_clearly(self):
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            with self.assertRaises(RuntimeError):
                launcher.available_port([busy.getsockname()[1]])

    def test_existing_project_is_reused_without_starting_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "launcher.json"
            state.write_text(json.dumps({"project": str(launcher.ROOT.resolve()), "port": 8503}), encoding="utf-8")
            with patch.object(launcher, "STATE", state), patch.object(launcher, "DATA", Path(tmp)), \
                 patch.object(launcher, "healthy", return_value=True), patch.object(launcher.subprocess, "Popen") as spawn:
                self.assertEqual(launcher.start_or_reuse(), (launcher.local_url(8503), True))
                spawn.assert_not_called()

    def test_unrelated_project_state_is_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "launcher.json"
            state.write_text(json.dumps({"project": "other-project", "port": 8501}), encoding="utf-8")
            with patch.object(launcher, "STATE", state), patch.object(launcher, "healthy", return_value=True):
                self.assertIsNone(launcher.existing_server())

    def test_corrupt_state_does_not_prevent_fresh_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "launcher.json"
            state.write_text("invalid json", encoding="utf-8")
            with patch.object(launcher, "STATE", state):
                self.assertIsNone(launcher.existing_server())
