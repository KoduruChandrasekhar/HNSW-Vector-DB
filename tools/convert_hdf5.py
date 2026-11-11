#!/usr/bin/env python3
"""Convert an ANN-Benchmarks HDF5 file into the simple binary files vdb reads.

    python tools/convert_hdf5.py --download sift             # -> data/sift/
    python tools/convert_hdf5.py data/glove-100-angular.hdf5 data/glove-100

Writes base.fbin, query.fbin and gt.ibin: a uint32 count, a uint32 dimension, then the
raw little-endian float32 (or int32) values. C++ loads them with one read, no HDF5.
"""

import argparse
import os
import struct
import sys
import urllib.request

import numpy as np

DATASETS = {
    "sift": "sift-128-euclidean.hdf5",
    "glove-100": "glove-100-angular.hdf5",
    "fashion-mnist": "fashion-mnist-784-euclidean.hdf5",
    "gist": "gist-960-euclidean.hdf5",
}
BASE_URL = "http://ann-benchmarks.com/"


def write_bin(path, array, dtype):
    a = np.ascontiguousarray(array, dtype=dtype)
    with open(path, "wb") as f:
        f.write(struct.pack("<II", a.shape[0], a.shape[1]))
        f.write(a.astype(a.dtype.newbyteorder("<")).tobytes())
    print(f"  {path}: {a.shape[0]} x {a.shape[1]}")


def download(name, data_dir):
    fname = DATASETS[name]
    path = os.path.join(data_dir, fname)
    if os.path.exists(path):
        return path
    os.makedirs(data_dir, exist_ok=True)
    url = BASE_URL + fname
    print(f"downloading {url}")
    # The CDN in front of ann-benchmarks.com rejects urllib's default User-Agent (403).
    req = urllib.request.Request(url, headers={"User-Agent": "vdb-convert/1.0 (+https://github.com)"})
    with urllib.request.urlopen(req, timeout=60) as r, open(path + ".part", "wb") as out:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r  {done * 100 // total}% of {total / 1e6:.0f} MB")
                sys.stdout.flush()
    os.replace(path + ".part", path)
    print()
    return path


def convert(hdf5_path, out_dir):
    import h5py

    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(hdf5_path, "r") as f:
        print(f"{hdf5_path}: distance={f.attrs.get('distance')}")
        write_bin(os.path.join(out_dir, "base.fbin"), f["train"][:], np.float32)
        write_bin(os.path.join(out_dir, "query.fbin"), f["test"][:], np.float32)
        write_bin(os.path.join(out_dir, "gt.ibin"), f["neighbors"][:], np.int32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hdf5", nargs="?", help="input .hdf5 file")
    ap.add_argument("out", nargs="?", help="output directory")
    ap.add_argument("--download", choices=sorted(DATASETS), help="fetch a dataset from ann-benchmarks.com")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    if args.download:
        path = download(args.download, args.data_dir)
        convert(path, os.path.join(args.data_dir, args.download))
    elif args.hdf5 and args.out:
        convert(args.hdf5, args.out)
    else:
        ap.error("give an input file and output directory, or --download NAME")


if __name__ == "__main__":
    main()
