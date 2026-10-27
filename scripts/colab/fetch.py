import tarfile
with tarfile.open("/content/deep/out_all.tar.gz", "w:gz") as tf:
    for d in ("out", "out_v2", "out_ablation"):
        tf.add(f"/content/deep/{d}", arcname=d)
print("packed")
