import os, subprocess
pid = open("/content/hold/train.pid").read().strip()
alive = subprocess.run(["kill", "-0", pid]).returncode == 0
print("alive:", alive, "| DONE:", os.path.exists("/content/hold/out/DONE"))
print("\n".join(open("/content/hold/train.log", errors="replace").read().splitlines()[-4:]))
