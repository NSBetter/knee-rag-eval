from __future__ import annotations
import csv, hashlib, json, shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/retrieval_benchmark_freeze_v1.json"
CORPUS = ROOT / "data/processed/gold_corpus/gold_corpus_v1_3.csv"

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        return [{k:(v or "").strip() for k,v in row.items()} for row in r]

def split_ids(value):
    return [x.strip() for x in (value or "").split("|") if x.strip()]

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = ROOT / cfg["source_path"]
    frozen = ROOT / cfg["frozen_path"]
    manifest_path = ROOT / cfg["manifest_path"]
    public_path = ROOT / cfg["public_manifest_path"]
    rows = [r for r in read_csv(source) if r["phase"] == cfg["scope"]]
    corpus = read_csv(CORPUS)
    chunk_source = {r["chunk_id"]: r["source_id"] for r in corpus}
    issues=[]; seen=set(); multi=no_gold=cross=0
    for r in rows:
        qid=r["query_id"]
        if qid in seen: issues.append(f"{qid}: duplicate_query_id")
        seen.add(qid)
        gold=set(split_ids(r["gold_chunk_ids"])); support=set(split_ids(r["supporting_chunk_ids"]))
        unknown=(gold|support)-set(chunk_source)
        if unknown: issues.append(f"{qid}: unknown_chunk_ids={sorted(unknown)}")
        if gold & support: issues.append(f"{qid}: gold_support_overlap={sorted(gold & support)}")
        sources={chunk_source[x] for x in gold if x in chunk_source}
        expected=set(split_ids(r["expected_source_ids"]))
        if sources and sources != expected: issues.append(f"{qid}: expected_sources={sorted(expected)} actual={sorted(sources)}")
        if r["evidence_scope"]=="multi_chunk":
            multi+=1
            if len(gold)<2: issues.append(f"{qid}: multi_chunk_requires_2_gold")
        if r["evidence_scope"]=="single_chunk" and len(gold)!=1: issues.append(f"{qid}: single_chunk_gold_count={len(gold)}")
        if r["evidence_scope"]=="no_gold":
            no_gold+=1
            if gold: issues.append(f"{qid}: no_gold_contains_gold")
        if len(sources)>1: cross+=1
    verified=sum(r["review_status"]=="verified" for r in rows)
    checks=[
        (len(rows)==cfg["required_selected_rows"], f"selected_rows={len(rows)}"),
        (verified==cfg["required_verified_rows"], f"verified_rows={verified}"),
        (multi>=cfg["minimum_multi_chunk_rows"], f"multi_chunk_rows={multi}"),
        (no_gold>=cfg["minimum_no_gold_rows"], f"no_gold_rows={no_gold}"),
        (cross>=cfg["minimum_cross_source_rows"], f"cross_source_rows={cross}"),
    ]
    issues += [msg for ok,msg in checks if not ok]
    if issues:
        print("Benchmark freeze blocked")
        for issue in issues: print("-", issue)
        raise SystemExit(1)
    frozen.parent.mkdir(parents=True, exist_ok=True)
    source_hash=sha256(source)
    if frozen.exists() and sha256(frozen)!=source_hash:
        raise SystemExit("Frozen snapshot already exists with a different hash; create a new version instead of overwriting it.")
    if not frozen.exists(): shutil.copy2(source, frozen)
    manifest={
        "benchmark_version":cfg["benchmark_version"], "scope":cfg["scope"],
        "source_path":cfg["source_path"], "frozen_path":cfg["frozen_path"],
        "sha256":sha256(frozen), "frozen_at_utc":datetime.now(timezone.utc).isoformat(),
        "selected_rows":len(rows), "verified_rows":verified,
        "answerable_rows":sum(r["answerability"]=="answerable" for r in rows),
        "unanswerable_rows":sum(r["answerability"]=="unanswerable" for r in rows),
        "single_chunk_rows":sum(r["evidence_scope"]=="single_chunk" for r in rows),
        "multi_chunk_rows":multi, "no_gold_rows":no_gold, "cross_source_rows":cross,
        "cross_source_policy":cfg["cross_source_policy"], "policy_note":cfg["policy_note"],
        "corpus_path":str(CORPUS.relative_to(ROOT)), "corpus_sha256":sha256(CORPUS)
    }
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    write_csv(public_path,[{k:manifest[k] for k in ["benchmark_version","scope","sha256","selected_rows","verified_rows","answerable_rows","unanswerable_rows","single_chunk_rows","multi_chunk_rows","no_gold_rows","cross_source_rows","cross_source_policy","corpus_sha256"]}])
    print("Retrieval benchmark frozen")
    print("Version:",manifest["benchmark_version"]); print("SHA-256:",manifest["sha256"])
    print("Frozen file:",frozen); print("Manifest:",manifest_path)

if __name__ == "__main__": main()
