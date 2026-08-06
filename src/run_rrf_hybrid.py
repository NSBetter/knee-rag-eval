from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/retrieval_rrf_v1.json"
MANIFEST=ROOT/"data/benchmark/frozen/retrieval_eval_v1_pilot_manifest.json"
BM25=ROOT/"data/processed/retrieval_runs/bm25_char_bigram_v1_results.csv"
DENSE=ROOT/"data/processed/retrieval_runs/qwen3_embedding_0_6b_dense_v1_results.csv"
LOCAL=ROOT/"data/processed/retrieval_runs/bm25_qwen3_rrf_v1_results.csv"
METRICS=ROOT/"docs/retrieval_rrf_v1_metrics.csv"
QUERY_METRICS=ROOT/"docs/retrieval_rrf_v1_query_metrics.csv"

def read_csv(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        return [{k:(v or "").strip() for k,v in r.items()} for r in csv.DictReader(f)]

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def split_ids(v): return {x.strip() for x in (v or "").split("|") if x.strip()}
def mean(v): return sum(v)/len(v) if v else 0.0

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def index(rows,depth):
    out={}
    for r in rows:
        if int(r["rank"])<=depth: out.setdefault(r["query_id"],{})[r["retrieved_chunk_id"]]=r
    return out

def main():
    cfg=json.loads(CONFIG.read_text(encoding="utf-8")); manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    frozen=ROOT/manifest["frozen_path"]
    if sha256(frozen)!=manifest["sha256"]: raise SystemExit("Frozen benchmark hash mismatch")
    bench={r["query_id"]:r for r in read_csv(frozen) if r["phase"]==manifest["scope"]}
    bm=index(read_csv(BM25),cfg["input_depth"]); de=index(read_csv(DENSE),cfg["input_depth"])
    if set(bm)!=set(bench) or set(de)!=set(bench): raise SystemExit("Rerun BM25 and Dense against the frozen benchmark first")
    local=[]; qm=[]; h1=[]; h3=[]; h5=[]; rec=[]; mrr=[]; rh1=[]; rh3=[]; rh5=[]; ans=unans=0
    for qid in sorted(bench):
        b=bench[qid]; gold=split_ids(b["gold_chunk_ids"]); support=split_ids(b["supporting_chunk_ids"]); relevant=gold|support
        candidates=[]
        for cid in set(bm[qid])|set(de[qid]):
            br=bm[qid].get(cid); dr=de[qid].get(cid); brank=int(br["rank"]) if br else None; drank=int(dr["rank"]) if dr else None
            score=(1/(cfg["rrf_k"]+brank) if brank else 0)+(1/(cfg["rrf_k"]+drank) if drank else 0)
            meta=br or dr
            candidates.append((cid,score,brank,drank,meta))
        candidates.sort(key=lambda x:(-x[1],min(r for r in [x[2],x[3]] if r is not None),x[0]))
        top=candidates[:cfg["output_top_k"]]; ids=[x[0] for x in top]
        fg=next((i for i,c in enumerate(ids,1) if c in gold),None); fr=next((i for i,c in enumerate(ids,1) if c in relevant),None)
        if b["answerability"]=="answerable":
            ans+=1; a1=float(bool(gold&set(ids[:1]))); a3=float(bool(gold&set(ids[:3]))); a5=float(bool(gold&set(ids[:5]))); ar=len(gold&set(ids[:5]))/len(gold); am=1/fg if fg else 0.0
            h1.append(a1);h3.append(a3);h5.append(a5);rec.append(ar);mrr.append(am);rh1.append(float(bool(relevant&set(ids[:1]))));rh3.append(float(bool(relevant&set(ids[:3]))));rh5.append(float(bool(relevant&set(ids[:5]))))
        else: unans+=1; a1=a3=a5=ar=am=""
        qm.append({"run_id":cfg["run_id"],"query_id":qid,"answerability":b["answerability"],"gold_count":len(gold),"support_count":len(support),"first_gold_rank":fg or "","first_relevant_rank":fr or "","hit_at_1":a1,"hit_at_3":a3,"hit_at_5":a5,"recall_at_5":round(ar,6) if ar!="" else "","top1_chunk_id":ids[0]})
        for rank,(cid,score,brank,drank,meta) in enumerate(top,1):
            local.append({"run_id":cfg["run_id"],"query_id":qid,"query":b["query"],"answerability":b["answerability"],"gold_chunk_ids":b["gold_chunk_ids"],"supporting_chunk_ids":b["supporting_chunk_ids"],"rank":rank,"retrieved_chunk_id":cid,"retrieved_source_id":meta.get("retrieved_source_id",""),"retrieved_title_path":meta.get("retrieved_title_path",""),"rrf_score":round(score,9),"bm25_rank":brank or "","dense_rank":drank or "","is_gold":int(cid in gold),"is_support":int(cid in support)})
    m={"run_id":cfg["run_id"],"benchmark_version":manifest["benchmark_version"],"retriever":"reciprocal_rank_fusion","rrf_k":cfg["rrf_k"],"input_depth":cfg["input_depth"],"output_top_k":cfg["output_top_k"],"evaluated_answerable_queries":ans,"reported_unanswerable_queries":unans,"hit_at_1":round(mean(h1),6),"hit_at_3":round(mean(h3),6),"hit_at_5":round(mean(h5),6),"recall_at_5":round(mean(rec),6),"mrr_at_10":round(mean(mrr),6),"relevant_hit_at_1":round(mean(rh1),6),"relevant_hit_at_3":round(mean(rh3),6),"relevant_hit_at_5":round(mean(rh5),6)}
    write_csv(LOCAL,local);write_csv(METRICS,[m]);write_csv(QUERY_METRICS,qm)
    print("RRF hybrid retrieval");print("="*88);print(f"Hit@1={m['hit_at_1']} Hit@3={m['hit_at_3']} Hit@5={m['hit_at_5']} Recall@5={m['recall_at_5']} MRR@10={m['mrr_at_10']}");print("Local results:",LOCAL);print("Public metrics:",METRICS);print("Public query metrics:",QUERY_METRICS);print("="*88)
if __name__=="__main__": main()
