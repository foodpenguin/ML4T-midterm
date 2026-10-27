import os, subprocess, tarfile
os.makedirs("/content/ml4t", exist_ok=True)
with tarfile.open("/content/deep/bundle.tar.gz") as tf:
    tf.extractall("/content/ml4t")
p = subprocess.Popen(["python", "scripts/run_deep.py", "train", "--data", "/content/deep", "--out", "/content/deep/out"],
                     cwd="/content/ml4t", stdout=open("/content/deep/train_stdout.log", "a"),
                     stderr=subprocess.STDOUT, start_new_session=True)
open("/content/deep/train.pid", "w").write(str(p.pid))
print("launched pid", p.pid)
