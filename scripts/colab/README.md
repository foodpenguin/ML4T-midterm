# Running rung 4 on Google Colab (A100)

These helpers drive `scripts/run_deep.py train` on a Colab VM through
[`google-colab-cli`](https://pypi.org/project/google-colab-cli/). Set `COLAB_BIN` if `colab` is not on
`PATH`; the repository root is derived from the scripts' location.

1. `python scripts/run_deep.py prepare` builds `artifacts/deep/{X,mask,static}.npy`,
   `meta.parquet` and `bundle.tar.gz` (src/, config/, setup.yaml, the driver).
2. `setup_vm.sh` creates the A100 VM and runs the uploads:
3. `upload.sh` uploads the small files. `X.npy` exceeds the upload limit, so it is split into
   30 MB parts (`artifacts/deep/parts/`) and `upload_parts.sh` uploads them and reassembles
   them on the VM, printing both SHA-256 hashes for comparison.
4. `run.sh launch.py` starts the grid, Optuna and ablation runs in sequence as a detached process
   (survives CLI disconnects); `wait_done.sh` polls `poll.py` every 2 minutes until all three are DONE.
5. `fetch_and_stop.sh` downloads all outputs and stops the VM only after the download,
   extraction and DONE checks succeed.
6. `python scripts/run_deep.py evaluate` writes `reports/deep.md`.

Training resumes from `out/tuning.csv` and the per-fold checkpoint in `out/predictions.parquet`
if the VM is interrupted.
