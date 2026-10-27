import os, subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
os.makedirs("/content/deep", exist_ok=True)
for f in sorted(os.listdir("/content/deep")):
    print(f, os.path.getsize(f"/content/deep/{f}"))
