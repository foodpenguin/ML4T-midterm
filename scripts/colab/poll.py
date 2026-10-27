import os, subprocess
RUNS = ("out", "out_v2", "out_ablation")
pid = open("/content/deep/train.pid").read().strip()
alive = subprocess.run(["kill", "-0", pid]).returncode == 0
done = [os.path.exists(f"/content/deep/{d}/DONE") for d in RUNS]
print("alive:", alive, "| DONE:", all(done), "| per run:", dict(zip(RUNS, done)))
for f in [f"/content/deep/{d}/train.log" for d in RUNS] + ["/content/deep/train_stdout.log"]:
    if os.path.exists(f):
        lines = open(f, encoding="utf-8", errors="replace").read().splitlines()
        print(f"--- {f} ({len(lines)} lines)")
        print("\n".join(lines[-3:]))
