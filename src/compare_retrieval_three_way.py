from __future__ import annotations
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FILES={
    "BM25":ROOT/"docs/retrieval_bm25_v1_metrics.csv",
    "Dense":ROOT/"docs/retrieval_dense_qwen3_v1_metrics.csv",
    "RRF":ROOT/"docs/retrieval_rrf_v1_metrics.csv",
}
QUERY_FILES={
    "BM25":ROOT/"docs/retrieval_bm25_v1_query_metrics.csv",
    "Dense":ROOT/"docs/retrieval_dense_qwen3_v1_query_metrics.csv",
    "RRF":ROOT/"docs/retrieval_rrf_v1_query_metrics.csv",
}
OUT=ROOT/"docs/retrieval_bm25_dense_rrf_v1.csv"

def read_one(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:return next(csv.DictReader(f))
def read_index(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:return {r["query_id"]:r for r in csv.DictReader(f)}
def rank(v): return int(float(v)) if v else 999

def main():
    print("\nThree-way retrieval metrics")
    print("="*88)
    for name,path in FILES.items():
        r=read_one(path)
        print(f"{name:<6} Hit@1={r['hit_at_1']} Hit@3={r['hit_at_3']} Hit@5={r['hit_at_5']} Recall@5={r['recall_at_5']} MRR@10={r['mrr_at_10']}")
    data={k:read_index(v) for k,v in QUERY_FILES.items()}
    rows=[]
    for qid in sorted(data["RRF"]):
        ranks={k:rank(data[k][qid].get("first_gold_rank","")) for k in data}
        best=min(ranks.values()); winners="|".join(k for k,v in ranks.items() if v==best)
        rows.append({
            "query_id":qid,"answerability":data["RRF"][qid].get("answerability",""),
            "bm25_first_gold_rank":data["BM25"][qid].get("first_gold_rank",""),
            "dense_first_gold_rank":data["Dense"][qid].get("first_gold_rank",""),
            "rrf_first_gold_rank":data["RRF"][qid].get("first_gold_rank",""),
            "winner":winners,
            "bm25_recall_at_5":data["BM25"][qid].get("recall_at_5",""),
            "dense_recall_at_5":data["Dense"][qid].get("recall_at_5",""),
            "rrf_recall_at_5":data["RRF"][qid].get("recall_at_5",""),
            "bm25_top1_chunk_id":data["BM25"][qid].get("top1_chunk_id",""),
            "dense_top1_chunk_id":data["Dense"][qid].get("top1_chunk_id",""),
            "rrf_top1_chunk_id":data["RRF"][qid].get("top1_chunk_id","")
        })
    with OUT.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print("Comparison CSV:",OUT);print("="*88)
if __name__=="__main__":main()
