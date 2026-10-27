# Running rung 4 on Google Colab (A100)

These helpers drive `scripts/run_deep.py train` on a Colab VM through
[`google-colab-cli`](https://pypi.org/project/google-colab-cli/). The shell scripts hard-code
this machine's WSL paths (`C=` points to the `colab` binary, `D=` to `artifacts/deep`); edit
those two lines elsewhere.

1. `python scripts/run_deep.py prepare` builds `artifacts/deep/{X,mask,static}.npy`,
   `meta.parquet` and `bundle.tar.gz` (src/, config/, setup.yaml, the driver).
2. `colab new -s ml4t-deep --gpu A100`
3. `upload.sh` uploads the small files. `X.npy` exceeds the upload limit, so it is split into
   30 MB parts (`artifacts/deep/parts/`) and `upload_parts.sh` uploads them and reassembles
   them on the VM, printing both SHA-256 hashes for comparison.
4. `run.sh launch.py` starts training as a detached process (survives CLI disconnects);
   `wait_done.sh` polls `poll.py` every 2 minutes until `out/DONE` exists.
5. `run.sh fetch.py`, then `colab download -s ml4t-deep /content/deep/out.tar.gz ...`,
   extract into `artifacts/deep/`, and `colab stop -s ml4t-deep`.
6. `python scripts/run_deep.py evaluate` writes `reports/deep.md`.

Training resumes from `out/tuning.csv` and the per-fold checkpoint in `out/predictions.parquet`
if the VM is interrupted.
