import hashlib, glob, os
parts = sorted(glob.glob("/content/deep/parts/X.part*"))
with open("/content/deep/X.npy", "wb") as out:
    for p in parts:
        out.write(open(p, "rb").read())
h = hashlib.sha256(open("/content/deep/X.npy", "rb").read()).hexdigest()
print(len(parts), "parts ->", os.path.getsize("/content/deep/X.npy"), "bytes, sha256", h)
