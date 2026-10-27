import os, shutil, subprocess, tarfile
shutil.rmtree("/content/ml4t", ignore_errors=True)
os.makedirs("/content/ml4t")
with tarfile.open("/content/hold/bundle.tar.gz") as tf:
    tf.extractall("/content/ml4t", filter="data")
cmd = "python scripts/run_deep.py holdout-train --data /content/hold --out /content/hold/out"
p = subprocess.Popen(["bash", "-c", cmd], cwd="/content/ml4t", stdout=open("/content/hold/train.log", "a"),
                     stderr=subprocess.STDOUT, start_new_session=True)
open("/content/hold/train.pid", "w").write(str(p.pid))
print("launched pid", p.pid)
