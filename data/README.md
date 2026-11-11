# data/

Datasets are not committed. Fetch and convert them with:

```bash
bench/prepare_data.sh                  # fashion-mnist, sift, glove-100
python tools/convert_hdf5.py --download gist   # optional, 3.6 GB
```

Each dataset becomes `data/<name>/base.fbin`, `query.fbin` and `gt.ibin` (uint32 count,
uint32 dimension, then raw little-endian float32 / int32). `data/sift-100k/` is the
cached subset used by the recall-regression test.
