import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def start():
    root = Path(__file__).resolve().parent.parent
    frontend = root / "frontend"
    if not (frontend / "node_modules" / "next").exists():
        raise SystemExit("请先在 frontend 执行 npm ci。")
    backend_port = int(os.getenv("MUSICTRACKER_PORT", "8100"))
    for port in [backend_port, 4321]:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise SystemExit(f"端口 {port} 已占用，未终止任何已有进程；请先关闭已有 MusicTracker 或调整后端端口。")
    processes = []
    try:
        processes.append(subprocess.Popen([sys.executable, "-m", "musictracker", "serve"], cwd=root, start_new_session=True))
        env = os.environ | {"MUSICTRACKER_BACKEND_URL": f"http://127.0.0.1:{backend_port}", "NEXT_TELEMETRY_DISABLED": "1"}
        processes.append(subprocess.Popen(["npm", "run", "dev"], cwd=frontend, env=env, start_new_session=True))
        print("MusicTracker：http://127.0.0.1:4321  ·  Ctrl+C 停止本次启动的服务", flush=True)
        while all(p.poll() is None for p in processes):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"进程 {process.pid} 尚未退出；未强制终止，请检查终端。", flush=True)
