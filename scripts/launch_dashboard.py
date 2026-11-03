"""Idempotent local launcher: reuse this project's server and avoid busy ports.

Only standard-library dependencies. No credentials are read or printed here.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BASE_PATH = "radar-" + hashlib.sha256(str(ROOT.resolve()).casefold().encode()).hexdigest()[:10]
STATE = DATA / "launcher.json"


def local_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/{BASE_PATH}"


def healthy(url: str) -> bool:
    try:
        # Never send loopback checks to a configured proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url + "/_stcore/health", timeout=1) as result:
            return result.status == 200 and result.read(32).strip() == b"ok"
    except (OSError, ValueError):
        return False


def existing_server() -> str | None:
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
        port = state["port"]
        if type(port) is not int or not 8501 <= port <= 8520:
            return None
        url = local_url(port)
        if state.get("project") == str(ROOT.resolve()) and healthy(url):
            return url
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def available_port(ports=range(8501, 8521)) -> int:
    for port in ports:
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available port between 8501 and 8520.")


@contextmanager
def launch_lock():
    """Serialize simultaneous double-clicks, including the readiness wait."""
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "launcher.lock").open("a+b") as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        deadline = time.monotonic() + 60
        while True:
            try:
                lock.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("Another launch is still in progress. Try again shortly.")
                time.sleep(.25)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_UN)


def start_or_reuse() -> tuple[str, bool]:
    with launch_lock():
        existing = existing_server()
        if existing:
            return existing, True
        port = available_port()
        url = local_url(port)
        command = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
                   "--server.address", "127.0.0.1", "--server.port", str(port),
                   "--server.headless", "true", "--server.baseUrlPath", BASE_PATH]
        options = {"cwd": ROOT, "stdin": subprocess.DEVNULL, "close_fds": True}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        with (DATA / "server.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(command, stdout=log, stderr=log, **options)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError("Server startup failed. See data/server.log for details.")
            if healthy(url):
                # Project-specific base path prevents reusing some other app's health endpoint.
                temporary = STATE.with_suffix(".tmp")
                temporary.write_text(json.dumps({"project": str(ROOT.resolve()), "port": port, "pid": child.pid}), encoding="utf-8")
                temporary.replace(STATE)
                return url, False
            time.sleep(.25)
        child.terminate()
        raise RuntimeError("Server did not become ready within 45 seconds. See data/server.log.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true", help="Verify startup without opening the browser")
    args = parser.parse_args()
    try:
        url, reused = start_or_reuse()
        print("Reusing the running dashboard." if reused else "Dashboard started successfully.", flush=True)
        print(url + "/", flush=True)
        if not args.no_browser:
            if not webbrowser.open(url + "/", new=0):
                print("Open the address above in your browser.")
        return 0
    except (OSError, RuntimeError) as exc:
        print(f"Unable to open dashboard: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
