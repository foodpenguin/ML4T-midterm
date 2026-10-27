import tarfile
with tarfile.open("/content/deep/out.tar.gz", "w:gz") as tf:
    tf.add("/content/deep/out", arcname="out")
print("packed")
