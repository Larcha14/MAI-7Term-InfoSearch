import os
import time
from typing import Any, Dict, List, Tuple
from pymongo import MongoClient
import yaml

def mongo_from_env(cfg: Dict[str, Any]):
    db_cfg = cfg.get("db", {})
    uri = os.getenv(db_cfg.get("uri_env", "MONGO_URI"))
    dbn = os.getenv(db_cfg.get("database_env", "MONGO_DB"))
    coln = os.getenv(db_cfg.get("collection_env", "MONGO_COLLECTION"))
    if not uri or not dbn or not coln:
        raise RuntimeError(f"Mongo env missing: uri={bool(uri)} db={bool(dbn)} col={bool(coln)}")
    client = MongoClient(uri)
    return client, client[dbn][coln]

def verify_mongo_collection(cfg: Dict[str, Any], sample_n: int = 30) -> None:
    client, col = mongo_from_env(cfg)
    try:
        total = col.estimated_document_count()
        print(f"VERIFY: total docs = {total}", flush=True)

        # count by source
        by_source = list(col.aggregate([
            {"$group": {"_id": "$source_name", "cnt": {"$sum": 1}}},
            {"$sort": {"cnt": -1}},
        ]))
        print("VERIFY: by source:", flush=True)
        for row in by_source:
            print(f"  - {row.get('_id')}: {row.get('cnt')}", flush=True)

        # html length stats
        html_stats = list(col.aggregate([
            {"$project": {"source_name": 1, "html_len": {"$strLenBytes": {"$ifNull": ["$html", ""]}}}},
            {"$group": {
                "_id": None,
                "min_len": {"$min": "$html_len"},
                "max_len": {"$max": "$html_len"},
                "avg_len": {"$avg": "$html_len"},
                "empty_html": {"$sum": {"$cond": [{"$eq": ["$html_len", 0]}, 1, 0]}},
                "short_html_lt_1000": {"$sum": {"$cond": [{"$lt": ["$html_len", 1000]}, 1, 0]}},
            }},
        ]))
        if html_stats:
            s = html_stats[0]
            print("VERIFY: html length stats:", flush=True)
            print(f"  min={int(s['min_len'])} bytes", flush=True)
            print(f"  max={int(s['max_len'])} bytes", flush=True)
            print(f"  avg={float(s['avg_len']):.1f} bytes", flush=True)
            print(f"  empty_html={int(s['empty_html'])}", flush=True)
            print(f"  short_html_lt_1000={int(s['short_html_lt_1000'])}", flush=True)

        # sample docs (latest first)
        print(f"VERIFY: sample {sample_n} docs (latest first):", flush=True)
        cur = col.find(
            {},
            {"_id": 0, "url_norm": 1, "source_name": 1, "fetched_at_unix": 1, "html": 1},
        ).sort("fetched_at_unix", -1).limit(sample_n)

        i = 0
        for doc in cur:
            i += 1
            html = doc.get("html") or ""
            print(
                f"  sample[{i}]: source={doc.get('source_name')} fetched_at={doc.get('fetched_at_unix')} "
                f"html_len={len(html)} url_norm={doc.get('url_norm')}",
                flush=True,
            )

        # required fields quick check
        missing_required = col.count_documents({
            "$or": [
                {"url_norm": {"$exists": False}},
                {"html": {"$exists": False}},
                {"source_name": {"$exists": False}},
                {"fetched_at_unix": {"$exists": False}},
            ]
        })
        print(f"VERIFY: docs missing required fields = {missing_required}", flush=True)

    finally:
        client.close()


def load_config() -> Dict[str, Any]:
    path = os.getenv("CONFIG_PATH", "/config/config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise RuntimeError("YAML root must be an object")
    return cfg

def main() -> int:
    cfg = load_config()
    n = int(os.getenv("VERIFY_SAMPLE_N", "30"))
    verify_mongo_collection(cfg, sample_n=n)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())