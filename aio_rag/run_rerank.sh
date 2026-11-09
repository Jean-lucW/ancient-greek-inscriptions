set -e
python rerank.py data bm25_char4 hybrid_char4+e5_raw
python retrieve.py data
