import tarfile
with tarfile.open("/content/hold/out.tar.gz", "w:gz") as tf:
    tf.add("/content/hold/out", arcname="deep_out")
print("packed")
