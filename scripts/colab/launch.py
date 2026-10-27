# Runs on the VM: extract the code bundle and start all registered rung-4 runs in sequence,
# detached from the CLI connection (grid -> optuna -> ablation built on the optuna config).
import os, shutil, subprocess, tarfile
shutil.rmtree("/content/ml4t", ignore_errors=True)
os.makedirs("/content/ml4t")
with tarfile.open("/content/deep/bundle.tar.gz") as tf:
    tf.extractall("/content/ml4t", filter="data")
run = "python scripts/run_deep.py train --data /content/deep"
cmd = (f"pip -q install optuna && "
       f"{run} --variant grid --out /content/deep/out && "
       f"{run} --variant optuna --out /content/deep/out_v2 && "
       f"{run} --variant ablation --out /content/deep/out_ablation --tuned-from /content/deep/out_v2")
p = subprocess.Popen(["bash", "-c", cmd], cwd="/content/ml4t", stdout=open("/content/deep/train_stdout.log", "a"),
                     stderr=subprocess.STDOUT, start_new_session=True)
open("/content/deep/train.pid", "w").write(str(p.pid))
print("launched pid", p.pid)
