#!/usr/bin/env python3
"""brainwave — local LLM server manager"""

import json, subprocess, sys, os, time, urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = json.load(open(os.path.join(SCRIPT_DIR, "config.json")))
LLAMA_BIN = CONFIG["llama_bin"]
MODELS_DIR = CONFIG["models_dir"]
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")


def health(port):
    try:
        r = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
        return json.loads(r.read()).get("status") == "ok"
    except:
        return False


def quit_server(port):
    try:
        urllib.request.urlopen(f"http://localhost:{port}/quit", timeout=2)
    except:
        pass


def start(name):
    srv = CONFIG["servers"].get(name)
    if not srv:
        print(f"Unknown server: {name}")
        print(f"Available: {', '.join(CONFIG['servers'].keys())}")
        return

    if health(srv["port"]):
        print(f"[{name}] already running on :{srv['port']}")
        return

    model_path = os.path.join(MODELS_DIR, srv["model"])
    if not os.path.exists(model_path):
        print(f"ERROR: model not found: {model_path}")
        return

    exe = os.path.join(LLAMA_BIN, "llama-server.exe")
    if not os.path.exists(exe):
        exe = os.path.join(LLAMA_BIN, "llama-server")

    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = open(os.path.join(LOG_DIR, f"{name}.log"), "w")

    cmd = [
        exe,
        "-m", model_path,
        "--port", str(srv["port"]),
        "-ngl", str(srv["gpu_layers"]),
        "-c", str(srv["ctx_size"]),
        "-fa", srv["flash_attn"],
    ]

    print(f"[{name}] starting on :{srv['port']} — {srv['description']}")
    print(f"  model: {srv['model'].split('/')[-1]}")

    proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
    print(f"  pid: {proc.pid}")

    for _ in range(30):
        if health(srv["port"]):
            print(f"  status: ok")
            return
        time.sleep(1)

    print(f"  WARNING: didn't respond in 30s, check logs/{name}.log")


def stop(name):
    srv = CONFIG["servers"].get(name)
    if not srv:
        print(f"Unknown server: {name}")
        return

    if health(srv["port"]):
        print(f"[{name}] stopping on :{srv['port']}")
        quit_server(srv["port"])
        print(f"  stopped")
    else:
        print(f"[{name}] not running on :{srv['port']}")


def status():
    print("brainwave status")
    print("=" * 60)
    print()

    for name, srv in CONFIG["servers"].items():
        running = health(srv["port"])
        state = "RUNNING" if running else "STOPPED"
        model = srv["model"].split("/")[-1]
        print(f"  {name:<10} :{srv['port']:<5}  {state:<8}  {srv['vram_mb']}MB  {model}")
        print(f"             {srv['description']}")
        print()

    # GPU
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        print("GPU")
        print("---")
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            print(f"  {parts[0]}  {parts[1]}/{parts[2]} MB  {parts[3]}°C")
    except:
        pass


def models():
    inv = json.load(open(os.path.join(SCRIPT_DIR, "models.json")))
    print(f"{'Model':<35} {'Size':>6}  {'Quant':<8}  {'GLSL':>4}  Notes")
    print("-" * 90)
    for m in sorted(inv["inventory"], key=lambda x: -x["size_gb"]):
        glsl = "yes" if m.get("glsl_trained") else ""
        print(f"  {m['name']:<33} {m['size_gb']:>5.1f}G  {m['quant']:<8}  {glsl:>4}  {m['notes'][:40]}")


def main():
    args = sys.argv[1:]
    cmd = args[0] if args else "status"
    target = args[1] if len(args) > 1 else None

    if cmd == "start":
        if target:
            start(target)
        else:
            for name in CONFIG["servers"]:
                start(name)
    elif cmd == "stop":
        if target:
            stop(target)
        else:
            for name in CONFIG["servers"]:
                stop(name)
    elif cmd == "status":
        status()
    elif cmd == "models":
        models()
    else:
        print("Usage: python bw.py [start|stop|status|models] [server_name]")


if __name__ == "__main__":
    main()
