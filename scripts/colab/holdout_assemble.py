import glob, hashlib, os
parts = sorted(glob.glob("/content/hold/parts/X.part*"))
with open("/content/hold/X.npy", "wb") as out:
    for p in parts:
        out.write(open(p, "rb").read())
print(len(parts), "parts, sha256", hashlib.sha256(open("/content/hold/X.npy", "rb").read()).hexdigest())
