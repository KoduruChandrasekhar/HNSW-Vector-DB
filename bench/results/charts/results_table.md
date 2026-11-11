| Dataset | System | Parameters | Recall@10 | QPS, 1 thread | p99 ms | Build s | Bytes per vector |
|---|---|---|---|---|---|---|---|
| glove-100 | vdb hnsw | M=16, ef=640, efc=200 | 0.9363 | 537 | 3.694 | 367.8 | 564 |
| glove-100 | hnswlib | M=16, ef=640, efc=200 | 0.9358 | 539 | 3.526 | 340.2 | 540 |
| glove-100 | faiss_hnsw | M=16, ef=640, efc=200 | 0.9318 | 415 | 4.976 | 428.7 | 528 |
| glove-100 | vdb flat |  | 1.0000 | 30 | 66.163 | 0.0 | 400 |
| sift | vdb hnsw | M=16, ef=80, efc=200 | 0.9743 | 3391 | 0.692 | 210.1 | 676 |
| sift | hnswlib | M=16, ef=80, efc=200 | 0.9743 | 2601 | 1.199 | 209.7 | 652 |
| sift | faiss_hnsw | M=16, ef=80, efc=200 | 0.9775 | 2285 | 1.282 | 231.7 | 640 |
| sift | vdb hnsw_sq8 | M=16, ef=80, efc=200, rerank=-1 | 0.9748 | 6153 | 0.289 | 105.8 | 292 |
| sift | vdb ivf_pq | nlist=4096, nprobe=64, pq_m=16, rerank=100 | 0.9551 | 1763 | 1.168 | 141.9 | 89 |
| sift | faiss_ivfpq | nlist=4096, nprobe=64, pq_m=16, rerank=100 | 0.9557 | 2504 | 1.399 | 86.1 | 26 |
| sift | vdb flat |  | 0.9991 | 22 | 80.904 | 0.0 | 512 |
| sift | faiss_flat |  | 0.9994 | 166 | 73.078 | 0.4 | 512 |
