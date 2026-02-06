import os
import re
from datetime import datetime
from typing import Any, Dict, Tuple, Optional, List

from pymongo import MongoClient, ASCENDING, UpdateOne
import yaml

from bs4 import BeautifulSoup, Comment
from urllib.parse import unquote

REMOVE_TAGS = ["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]
WIKI_CUT_SECTIONS = {"примечания", "ссылки", "литература", "см. также", "источники"}

_ws_re = re.compile(r"\s+")
_ctrl_re = re.compile(r"[\u0000-\u001f\u007f]+")

def mongo_from_env(cfg: Dict[str, Any]) -> Tuple[MongoClient, Any, Any]:
    db_cfg = cfg.get("db", {})
    uri = os.getenv(db_cfg.get("uri_env", None))
    dbn = os.getenv(db_cfg.get("database_env", None))
    coln = os.getenv(db_cfg.get("collection_env", None))
    new_col = os.getenv(db_cfg.get("clean_collection_env",None))

    if not uri or not dbn or not coln or not new_col:
        raise RuntimeError(f"Mongo env missing: uri={bool(uri)} db={bool(dbn)} col={bool(coln)} clean_col={bool(new_col)}")

    client = MongoClient(uri)
    return client, client[dbn][coln], client[dbn][new_col]


def load_config() -> Dict[str, Any]:
    path = os.getenv("CONFIG_PATH", "/config/config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise RuntimeError("YAML root must be an object")
    return cfg


def _norm_text(s: str) -> str:
    if not s:
        return ""
    s = _ctrl_re.sub(" ", s)
    s = _ws_re.sub(" ", s).strip()
    return s



def clean_html(html: str) -> Tuple[Any, Any]:
    soup = BeautifulSoup(html or "", "lxml")

    h1 = ""
    h1_tag = soup.select_one("#firstHeading") or soup.find("h1")
    if h1_tag:
        h1 = _norm_text(h1_tag.get_text(" ", strip=True))
    if not h1 and soup.title:
        h1 = _norm_text(soup.title.get_text(" ", strip=True))

    mw = soup.select_one("#mw-content-text")
    if mw:

        root = mw.select_one(".prp-pages-output") \
               or mw.select_one(".mw-parser-output") \
               or mw
    else:
        root = soup.body or soup

    for sel in [
        "script, style, noscript",
        "#toc",
        ".mw-editsection",
        "sup.reference",
        ".reflist, ol.references",
        ".navbox, .vertical-navbox",
        "#catlinks",
        ".metadata",
        "#siteSub, #contentSub, #mw-content-subtitle",
        ".mw-indicators",
    ]:
        for n in root.select(sel):
            n.decompose()

    for tag in root.find_all(REMOVE_TAGS):
        tag.decompose()

    for c in root.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()

    for h2 in root.select("h2"):
        txt = _norm_text(h2.get_text(" ", strip=True)).lower()
        if any(name in txt for name in WIKI_CUT_SECTIONS):
            sib = h2
            while sib is not None:
                nxt = sib.find_next_sibling()
                sib.decompose()
                sib = nxt
            break

    text = root.get_text(separator=" ")
    text = _ws_re.sub(" ", text).strip()

    text = re.sub(r"\[\s*\d+\s*\]", " ", text)
    text = _ws_re.sub(" ", text).strip()

    return h1, text

def main() -> int:
    try:
        cfg = load_config()
        client, src, dst = mongo_from_env(cfg)

        dst.create_index([("url_norm", ASCENDING)], unique=True)
        dst.create_index([("source_name", ASCENDING)])
        dst.create_index([("src_id", ASCENDING)], unique=True)

        cur = src.find(
            {"url_norm": {"$exists": True}},
            {"_id": 1, "url_norm": 1, "source_name": 1, "html": 1}
        ).batch_size(200)

        processed = 0
        skipped = 0

        print("Total documents with url_norm:", src.count_documents({"url_norm": {"$exists": True}}))



        for doc in cur:
            try:
                url_norm = doc.get("url_norm")
                if not url_norm:
                    skipped += 1
                    continue

                html = doc.get("html") or ""
                if not html:
                    skipped += 1
                    continue

                h1, text = clean_html(html)

                dst.update_one(
                    {"url_norm": url_norm},
                    {"$set": {
                        "url_norm": url_norm,
                        "url_clean": unquote(url_norm),
                        "source_name": doc.get("source_name"),
                        "src_id": doc.get("_id"),
                        "clean_text": text,
                        "title": h1
                    }},
                    upsert=True
                )

                processed += 1
                if processed % 200 == 0:
                    print(f"cleaned: {processed} (skipped: {skipped})")

            except Exception as e:
                print(f"Error processing doc {doc.get('_id')}: {e}")
                skipped += 1

        print(f"DONE. cleaned: {processed}, skipped: {skipped}")

    except Exception as e:
        print(f"Failed to update {url_norm}: {e}")

    finally:
        client.close()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
