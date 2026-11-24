"""List / download from the BASKET-Multiview FileBrowser public share.

usage:
  share.py ls  <path> [depth]                       sizes of each child of <path>
  share.py get <path> <dest_dir> [--skip a,b,...]   download, skipping dirs named a,b
Downloads are resumable: files already present with the right size are skipped.
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://datastore.shannon.humansensing.cs.cmu.edu/api/public"
HASH = "3gdyvt0N"


def q(p):
    return urllib.parse.quote(p.strip("/"))


def listing(p):
    with urllib.request.urlopen("%s/share/%s/%s" % (BASE, HASH, q(p)), timeout=120) as r:
        return json.load(r)


def walk(p, skip=()):
    for it in listing(p).get("items", []):
        sub = p.strip("/") + "/" + it["name"]
        if it["isDir"]:
            if it["name"] in skip:
                continue
            yield from walk(sub, skip)
        else:
            yield sub, it["size"]


def du(p, depth):
    tot = 0
    for it in listing(p).get("items", []):
        sub = p.strip("/") + "/" + it["name"]
        s = sum(sz for _, sz in walk(sub)) if it["isDir"] else it["size"]
        tot += s
        if depth > 0:
            print("%9.2f GB  %s%s" % (s / 1e9, sub, "/" if it["isDir"] else ""), flush=True)
    return tot


def fetch(args):
    path, size, dest = args
    out = os.path.join(dest, path)
    if os.path.exists(out) and os.path.getsize(out) == size:
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for attempt in range(3):
        try:
            urllib.request.urlretrieve("%s/dl/%s/%s" % (BASE, HASH, q(path)), out + ".part")
            os.replace(out + ".part", out)
            return size
        except Exception as e:
            err = e
    print("FAILED", path, err, flush=True)
    return 0


if __name__ == "__main__":
    cmd, p = sys.argv[1], sys.argv[2]
    if cmd == "ls":
        depth = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        print("%9.2f GB  TOTAL %s" % (du(p, depth) / 1e9, p), flush=True)
    elif cmd == "get":
        dest = sys.argv[3]
        skip = tuple(sys.argv[sys.argv.index("--skip") + 1].split(",")) if "--skip" in sys.argv else ()
        files = list(walk(p, skip))
        print("%d files, %.2f GB" % (len(files), sum(s for _, s in files) / 1e9), flush=True)
        done = 0
        with ThreadPoolExecutor(16) as ex:
            for i, n in enumerate(ex.map(fetch, [(f, s, dest) for f, s in files])):
                done += n
                if i % 500 == 0:
                    print("%d/%d files, %.2f GB new" % (i, len(files), done / 1e9), flush=True)
        print("DONE", p, flush=True)
