import os, subprocess
pid = open("/content/deep/train.pid").read().strip() if os.path.exists("/content/deep/train.pid") else None
alive = pid is not None and subprocess.run(["kill", "-0", pid]).returncode == 0
print("alive:", alive, "| DONE:", os.path.exists("/content/deep/out/DONE"))
for f in ("/content/deep/out/train.log", "/content/deep/train_stdout.log"):
    if os.path.exists(f):
        lines = open(f, encoding="utf-8", errors="replace").read().splitlines()
        print(f"--- {f} ({len(lines)} lines)")
        print("\n".join(lines[-8:]))
