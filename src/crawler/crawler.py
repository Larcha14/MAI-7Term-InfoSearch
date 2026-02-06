import asyncio
import hashlib
import json
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from urllib.parse import urlsplit, urlunsplit, quote, unquote
import logging
from logging.handlers import QueueHandler, QueueListener
import queue
from pathlib import Path
import traceback

import aiohttp
import yaml
from pymongo import MongoClient, ReturnDocument
from pymongo.collection import Collection
import aiohttp



logger = logging.getLogger("crawler")

# ------------------------- small utils -------------------------

def require_obj(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = d.get(key)
    if not isinstance(v, dict):
        raise RuntimeError(f"Missing or invalid config.{key} (must be an object)")
    return v

def require(d: Dict[str, Any], key: str, t) -> Any:
    if key not in d:
        raise RuntimeError(f"Missing config.{key}")
    v = d[key]
    if not isinstance(v, t):
        raise RuntimeError(f"Invalid config.{key} type (expected {t}, got {type(v)})")
    return v

def getenv_required(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

def _decode_bytes(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    return str(x)

def now_unix() -> int:
    return int(time.time())


# ------------------------- config dataclasses -------------------------

@dataclass(frozen=True)
class DbCfg:
    uri_env: str
    database_env: str
    collection_env: str

@dataclass(frozen=True)
class RecrawlCfg:
    enabled: bool
    revisit_after_hours: float
    only_if_changed: bool
    use_etag: bool
    use_last_modified: bool
    use_content_hash: bool

@dataclass(frozen=True)
class LogicCfg:
    delay_sec: float
    concurrency: int
    per_host_concurrency: int
    user_agent: str
    max_unique_docs: int
    resume: bool
    checkpoint_file: str
    checkpoint_flush_every: int
    recrawl: RecrawlCfg


@dataclass(frozen=True)
class HttpCfg:
    timeout_connect: float
    timeout_read: float
    retries: int
    backoff_sec: float
    max_redirects: int
    max_response_bytes: int

@dataclass
class RetryableHttpError(Exception):
    status: int
    url: str
    host: str
    retry_after_sec: float

    def __str__(self) -> str:
        return f"RetryableHttpError(status={self.status}, host={self.host}, wait={self.retry_after_sec}s, url={self.url})"


@dataclass(frozen=True)
class NormCfg:
    strip_fragment: bool = True
    drop_trailing_slash: bool = True
    lowercase_host: bool = True
    default_scheme: str = "https"

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> "NormCfg":
        n = require_obj(cfg, "normalization")
        return NormCfg(
            strip_fragment=bool(n.get("strip_fragment", True)),
            drop_trailing_slash=bool(n.get("drop_trailing_slash", True)),
            lowercase_host=bool(n.get("lowercase_host", True)),
            default_scheme=str(n.get("default_scheme", "https")),
        )


@dataclass(frozen=True)
class SourceCfg:
    name: str
    allowed_domains: List[str]

@dataclass(frozen=True)
class SeedsCfg:
    file: str
    format: str
    has_header: bool
    shuffle: bool
    max_per_source: int

@dataclass(frozen=True)
class CrawlerCfg:
    db: DbCfg
    logic: LogicCfg
    http: HttpCfg
    norm: NormCfg
    sources: List[SourceCfg]
    seeds: SeedsCfg


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise RuntimeError("YAML root must be an object")
    return cfg


def parse_cfg(cfg: Dict[str, Any]) -> CrawlerCfg:
    db = require_obj(cfg, "db")
    logic = require_obj(cfg, "logic")
    norm_cfg = NormCfg.from_cfg(cfg)
    sources_raw = require(cfg, "sources", list)
    seeds = require_obj(cfg, "seeds")

    # db
    db_cfg = DbCfg(
        uri_env=require(db, "uri_env", str),
        database_env=require(db, "database_env", str),
        collection_env=require(db, "collection_env", str),
    )

    # recrawl
    rec = require_obj(logic, "recrawl")
    cd = require_obj(rec, "change_detection")
    rec_cfg = RecrawlCfg(
        enabled=bool(rec.get("enabled", False)),
        revisit_after_hours=float(rec.get("revisit_after_hours", 168)),
        only_if_changed=bool(rec.get("only_if_changed", True)),
        use_etag=bool(cd.get("use_etag", True)),
        use_last_modified=bool(cd.get("use_last_modified", True)),
        use_content_hash=bool(cd.get("use_content_hash", True)),
    )

    # checkpoint (optional section)
    chk = logic.get("checkpoint") or {}
    if chk and not isinstance(chk, dict):
        raise RuntimeError("config.logic.checkpoint must be an object if provided")

    logic_cfg = LogicCfg(
        delay_sec=float(logic.get("delay_sec", 0.3)),
        concurrency=int(logic.get("concurrency", 8)),
        per_host_concurrency=int(logic.get("per_host_concurrency", 2)),
        user_agent=str(logic.get("user_agent", "crawler")),
        max_unique_docs=int(logic.get("max_unique_docs", 32000)),
        resume=bool(logic.get("resume", True)),
        checkpoint_file=str(chk.get("state_file", "/data/logs/crawler_state.json")),
        checkpoint_flush_every=int(chk.get("flush_every_n", 50)),
        recrawl=rec_cfg,
    )

    # http
    http = require_obj(cfg, "http")

    timeout = http.get("timeout_sec") or {}
    if timeout and not isinstance(timeout, dict):
        raise RuntimeError("config.http.timeout_sec must be an object")

    retries = http.get("retries") or {}
    if retries and not isinstance(retries, dict):
        raise RuntimeError("config.http.retries must be an object")

    max_mb = float(http.get("max_response_mb", 25))

    http_cfg = HttpCfg(
        timeout_connect=float(timeout.get("connect", 10)),
        timeout_read=float(timeout.get("read", 60)),
        retries=int(retries.get("count", 3)),
        backoff_sec=float(retries.get("backoff_sec", 2)),
        max_redirects=int(http.get("max_redirects", 5)),
        max_response_bytes=int(max_mb * 1024 * 1024),
)


    # sources
    sources: List[SourceCfg] = []
    for s in sources_raw:
        if not isinstance(s, dict):
            raise RuntimeError("config.sources[] must be objects")
        name = str(s.get("name", "")).strip()
        if not name:
            raise RuntimeError("config.sources[].name is required")
        allowed = s.get("allowed_domains") or []
        if not isinstance(allowed, list) or not all(isinstance(x, str) for x in allowed):
            raise RuntimeError(f"config.sources[{name}].allowed_domains must be a list of strings")
        sources.append(SourceCfg(name=name, allowed_domains=[x.strip().lower() for x in allowed if x.strip()]))

    if not sources:
        raise RuntimeError("config.sources must contain at least one source")

    # seeds
    seeds_cfg = SeedsCfg(
        file=require(seeds, "file", str),
        format=str(seeds.get("format", "tsv")).lower(),
        has_header=bool(seeds.get("has_header", True)),
        shuffle=bool(seeds.get("shuffle", False)),
        max_per_source=int(seeds.get("max_per_source", 10**9)),
    )

    if seeds_cfg.format != "tsv":
        raise RuntimeError("Only seeds.format=tsv is supported in this implementation")

    return CrawlerCfg(
        db=db_cfg,
        logic=logic_cfg,
        http=http_cfg,
        norm=norm_cfg,
        sources=sources,
        seeds=seeds_cfg,
    )


# ------------------------- url normalization + domain checks -------------------------

def normalize_url(url: str, n: NormCfg) -> str:
    url = (url or "").strip()
    if not url:
        return ""

    parts = urlsplit(url)

    scheme = parts.scheme or n.default_scheme
    netloc = parts.netloc
    path = parts.path or "/"
    query = parts.query
    fragment = "" if n.strip_fragment else parts.fragment

    if not netloc and parts.path:
        # handle schemeless like //host/path or host/path
        if url.startswith("//"):
            parts2 = urlsplit(n.default_scheme + ":" + url)
            scheme = parts2.scheme
            netloc = parts2.netloc
            path = parts2.path or "/"
            query = parts2.query
            fragment = "" if n.strip_fragment else parts2.fragment
        else:
            # "host/path" is ambiguous; treat as path
            pass

    if n.lowercase_host and netloc:
        if "@" in netloc:
            auth, hostport = netloc.split("@", 1)
            netloc = auth + "@" + hostport.lower()
        else:
            netloc = netloc.lower()

    if n.drop_trailing_slash and path != "/" and path.endswith("/"):
        path = path[:-1]

    return urlunsplit((scheme, netloc, path, query, fragment))


def host_from_url(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except Exception:
        return ""

def host_allowed(host: str, allowed_domains: List[str]) -> bool:
    host = (host or "").lower()
    if not host:
        return False
    for d in allowed_domains:
        d = d.lower()
        if host == d or host.endswith("." + d):
            return True
    return False


# ------------------------- seeds loading -------------------------

@dataclass(frozen=True)
class SeedRow:
    url: str
    source: str

def read_seeds_tsv(path: str, has_header: bool) -> List[SeedRow]:
    rows: List[SeedRow] = []
    with open(path, "r", encoding="utf-8") as f:
        if has_header:
            _ = f.readline()
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            # expected columns include url and source (last)
            if len(parts) < 2:
                continue
            url = parts[0].strip()
            source = parts[-1].strip()
            if url and source:
                rows.append(SeedRow(url=url, source=source))
    return rows



# ------------------------- checkpoint state -------------------------

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

@dataclass
@dataclass
class State:
    shuffle_seed: int
    next_idx: int
    processed: int
    ok: int
    unchanged: int
    failed: int
    by_source: Dict[str, int] = field(default_factory=dict)
    retry_idx: List[int] = field(default_factory=list)
    saved_at_unix: int = 0

def load_state(path: str) -> Optional[State]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return State(
            shuffle_seed=int(d.get("shuffle_seed", 0)),
            next_idx=int(d.get("next_idx", 0)),
            processed=int(d.get("processed", 0)),
            ok=int(d.get("ok", 0)),
            unchanged=int(d.get("unchanged", 0)),
            failed=int(d.get("failed", 0)),
            by_source=dict(d.get("by_source", {})),
        )
    except FileNotFoundError:
        return None
    except Exception as e:
        raise RuntimeError(f"Cannot read state file '{path}': {e}")

def save_state(path: str, st: State) -> None:
    ensure_parent_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {
                "shuffle_seed": st.shuffle_seed,
                "next_idx": st.next_idx,
                "processed": st.processed,
                "ok": st.ok,
                "unchanged": st.unchanged,
                "failed": st.failed,
                "by_source": st.by_source,
                "saved_at_unix": now_unix(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    os.replace(tmp, path)


# ------------------------- mongo -------------------------

def mongo_collection(cfg: CrawlerCfg) -> Collection:
    uri = getenv_required(cfg.db.uri_env)
    dbn = getenv_required(cfg.db.database_env)
    coln = getenv_required(cfg.db.collection_env)

    client = MongoClient(uri)
    col = client[dbn][coln]

    # indexes
    col.create_index([("url_norm", 1)], unique=True)
    col.create_index([("source_name", 1)])
    col.create_index([("fetched_at_unix", 1)])
    return col


# ------------------------- host limiter (global + per-host + delay) -------------------------

class HostLimiter:
    def __init__(self, total: int, per_host: int, delay_sec: float):
        self.total = asyncio.Semaphore(max(1, total))
        self.per_host = max(1, per_host)
        self.delay_sec = max(0.0, delay_sec)

        self._host_sem: Dict[str, asyncio.Semaphore] = {}
        self._host_lock: Dict[str, asyncio.Lock] = {}
        self._next_time: Dict[str, float] = {}

    def _sem(self, host: str) -> asyncio.Semaphore:
        if host not in self._host_sem:
            self._host_sem[host] = asyncio.Semaphore(self.per_host)
        return self._host_sem[host]

    def _lock(self, host: str) -> asyncio.Lock:
        if host not in self._host_lock:
            self._host_lock[host] = asyncio.Lock()
        return self._host_lock[host]

    async def acquire(self, host: str):
        await self.total.acquire()
        await self._sem(host).acquire()

        if self.delay_sec > 0:
            async with self._lock(host):
                now = time.monotonic()
                nt = self._next_time.get(host, 0.0)
                wait = nt - now
                if wait > 0:
                    await asyncio.sleep(wait)
                self._next_time[host] = time.monotonic() + self.delay_sec

    def release(self, host: str):
        self._sem(host).release()
        self.total.release()


# ------------------------- fetch + recrawl -------------------------

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ResponseTooLarge(RuntimeError):
    pass

async def fetch_html(
    session: aiohttp.ClientSession,
    url: str,
    http_cfg,
    headers: Dict[str, str],
) -> Tuple[int, Dict[str, str], bytes]:
    """
    Returns: (status, headers_lower, body_bytes)
    """
    timeout = aiohttp.ClientTimeout(
        total=None,
        sock_connect=float(http_cfg.timeout_connect),
        sock_read=float(http_cfg.timeout_read),
    )

    max_bytes = int(http_cfg.max_response_bytes * 1024 * 1024)

    async with session.get(
        url,
        headers=headers,
        timeout=timeout,
        allow_redirects=True,
        max_redirects=int(http_cfg.max_redirects),
    ) as resp:
        status = resp.status
        rh = {k.lower(): v for k, v in resp.headers.items()}

        # Fast oversize check (if Content-Length exists)
        cl = rh.get("content-length")
        if cl and cl.isdigit():
            if int(cl) > max_bytes:
                raise ResponseTooLarge(f"Response too large: content-length={cl} > max_bytes={max_bytes}")

        # Stream read with hard limit
        buf = bytearray()
        async for chunk in resp.content.iter_chunked(64 * 1024):
            buf.extend(chunk)
            if len(buf) > max_bytes:
                raise ResponseTooLarge(f"Response too large: read={len(buf)} > max_bytes={max_bytes}")

        return status, rh, bytes(buf)
    

async def fetch_with_retries(
    session: aiohttp.ClientSession,
    url: str,
    http: HttpCfg,
    headers: Dict[str, str],
) -> Tuple[int, Dict[str, str], bytes]:
    last_err: Optional[Exception] = None
    for attempt in range(1, http.retries + 2):
        try:
            return await fetch_html(session, url, http, headers)
        except aiohttp.ClientResponseError as e:
            # don't retry on most 4xx
            if 400 <= e.status < 500 and e.status != 429:
                raise
            last_err = e
        except Exception as e:
            last_err = e

        if attempt <= http.retries:
            await asyncio.sleep(http.backoff_sec * attempt)

    raise RuntimeError(f"Fetch failed after retries. Last error: {last_err}")


async def process_one(
    cfg: CrawlerCfg,
    col: Collection,
    limiter: HostLimiter,
    session: aiohttp.ClientSession,
    seed: SeedRow,
) -> Tuple[bool, bool]:
    # returns: (ok, unchanged)
    src_map = {s.name: s for s in cfg.sources}
    if seed.source not in src_map:
        return (False, False)

    allowed = src_map[seed.source].allowed_domains
    url0 = seed.url
    url_norm = normalize_url(url0, cfg.norm)
    if not url_norm:
        return (False, False)

    host = host_from_url(url_norm)
    if not host_allowed(host, allowed):
        return (False, False)

    # DB lookup
    existing = col.find_one({"url_norm": url_norm}, projection={"_id": 0, "etag": 1, "last_modified": 1, "content_hash": 1, "fetched_at_unix": 1})
    due = True
    if existing and cfg.logic.recrawl.enabled:
        age_sec = now_unix() - int(existing.get("fetched_at_unix", 0) or 0)
        due = age_sec >= int(cfg.logic.recrawl.revisit_after_hours * 3600)

    if existing and cfg.logic.recrawl.enabled and not due:
        # not time yet
        return (True, True)

    headers = {
        "User-Agent": cfg.logic.user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    if existing and cfg.logic.recrawl.enabled and cfg.logic.recrawl.only_if_changed:
        if cfg.logic.recrawl.use_etag and existing.get("etag"):
            headers["If-None-Match"] = str(existing["etag"])
        if cfg.logic.recrawl.use_last_modified and existing.get("last_modified"):
            headers["If-Modified-Since"] = str(existing["last_modified"])

    await limiter.acquire(host)
    try:
        status, rh, body = await fetch_with_retries(session, url_norm, cfg.http, headers)
    finally:
        limiter.release(host)

    fetched_at = now_unix()

    if status == 429:
        ra = rh.get("retry-after")
        wait_s = 15.0
        if ra:
            try:
                wait_s = float(ra)
            except ValueError:
                wait_s = 15.0

        # небольшой jitter, чтобы воркеры не просыпались строем
        wait_s += random.uniform(0.0, 1.0)

        host = (urlsplit(url_norm).hostname or "").lower()
        raise RetryableHttpError(status=429, url=url_norm, host=host, retry_after_sec=wait_s)

    # 304 = not modified
    if status == 304:
        col.find_one_and_update(
            {"url_norm": url_norm},
            {"$set": {"fetched_at_unix": fetched_at}},
            return_document=ReturnDocument.AFTER,
        )
        return (True, True)

    if status != 200:
        logger.warning(f"HTTP_FAIL: source={seed.source} status={status} url={url_norm}")
        return (False, False)

    # final URL host check (after redirects)
    final_url = str(session._response.url) if hasattr(session, "_response") else url_norm  # safe fallback
    # aiohttp doesn't expose final url here cleanly without passing response; skip hard check

    etag = rh.get("etag")
    last_modified = rh.get("last-modified")

    content_hash = sha256_hex(body) if cfg.logic.recrawl.use_content_hash else None

    if existing and cfg.logic.recrawl.enabled and cfg.logic.recrawl.only_if_changed and cfg.logic.recrawl.use_content_hash:
        if existing.get("content_hash") and existing["content_hash"] == content_hash:
            col.find_one_and_update(
                {"url_norm": url_norm},
                {"$set": {"fetched_at_unix": fetched_at}},
                return_document=ReturnDocument.AFTER,
            )
            return (True, True)

    doc = {
        "url": url_norm,
        "url_norm": url_norm,
        "html": body.decode("utf-8", "replace"),
        "source_name": seed.source,
        "fetched_at_unix": fetched_at,
    }
    if etag:
        doc["etag"] = etag
    if last_modified:
        doc["last_modified"] = last_modified
    if content_hash:
        doc["content_hash"] = content_hash

    col.update_one({"url_norm": url_norm}, {"$set": doc}, upsert=True)
    return (True, False)


# ------------------------- runner -------------------------


def setup_file_logger(path: str) -> tuple[logging.Logger, QueueListener]:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    q = queue.Queue()
    handler = logging.FileHandler(path, encoding="utf-8")
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    handler.setFormatter(formatter)

    listener = QueueListener(q, handler)
    listener.start()

    # logger = logging.getLogger("crawler")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(QueueHandler(q))
    logger.propagate = False

    return logger, listener



async def run_crawler(cfg) -> int:
    col = mongo_collection(cfg)

    # ---- logger (file) ----
    log_path = getattr(cfg.logic, "error_log_file", "/data/logs/crawler_errors.log")
    logger, listener = setup_file_logger(log_path)

    # ---- load seeds (TSV) ----
    seeds = read_seeds_tsv(cfg.seeds.file, cfg.seeds.has_header)

    debug_limit = int(os.getenv("DEBUG_LIMIT", "0") or "0")
    if debug_limit > 0:
        seeds = seeds[:debug_limit]

    if cfg.logic.max_unique_docs and cfg.logic.max_unique_docs > 0:
        seeds = seeds[: cfg.logic.max_unique_docs]

    # ---- state (resume) ----
    if cfg.logic.resume:
        st = load_state(cfg.logic.checkpoint_file)
        if st is None:
            st = State(
                shuffle_seed=0,
                next_idx=0,
                processed=0,
                ok=0,
                unchanged=0,
                failed=0,
                by_source={},
                saved_at_unix=0,
                retry_idx=[],
            )
            save_state(cfg.logic.checkpoint_file, st)
    else:
        st = State(
            shuffle_seed=0,
            next_idx=0,
            processed=0,
            ok=0,
            unchanged=0,
            failed=0,
            by_source={},
            saved_at_unix=0,
        )

    # deterministic shuffle for resume
    if cfg.seeds.shuffle:
        if getattr(st, "shuffle_seed", 0) in (0, None):
            st.shuffle_seed = random.randint(1, 2**31 - 1)
            if cfg.logic.resume:
                save_state(cfg.logic.checkpoint_file, st)
        rnd = random.Random(st.shuffle_seed)
        rnd.shuffle(seeds)

    total = len(seeds)
    if st.next_idx >= total:
        print(f"OK: nothing to do. next_idx={st.next_idx}, total={total}", flush=True)
        listener.stop()
        return 0

    print(f"START: seeds_total={total} next_idx={st.next_idx} shuffle={cfg.seeds.shuffle}", flush=True)

    limiter = HostLimiter(cfg.logic.concurrency, cfg.logic.per_host_concurrency, cfg.logic.delay_sec)

    stop_event = asyncio.Event()

    def _stop(*_):
        stop_event.set()
        print("STOP requested (Ctrl+C / SIGTERM).", flush=True)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass

    # ---- progress bookkeeping ----
    completed: Set[int] = set()
    lock = asyncio.Lock()
    flush_every = max(1, int(getattr(cfg.logic, "checkpoint_flush_every", 25)))
    progress_every = max(1, int(getattr(cfg.logic, "progress_every", 200)))

    # ---- 429 retry controls ----
    # сколько раз максимум пробуем один idx при 429, чтобы не зависнуть навечно
    max_429_retries = int(getattr(getattr(cfg.http, "retries", None), "count", 3))

    retry429_count: Dict[int, int] = {}
    host_cooldown_until: Dict[str, float] = {}
    host_lock = asyncio.Lock()

    async def set_host_cooldown(host: str, wait_s: float) -> None:
        if not host:
            return
        until = time.monotonic() + max(0.0, float(wait_s))
        async with host_lock:
            prev = host_cooldown_until.get(host, 0.0)
            if until > prev:
                host_cooldown_until[host] = until

    async def wait_host_if_cooled(host: str) -> None:
        if not host:
            return
        async with host_lock:
            until = host_cooldown_until.get(host, 0.0)
        now = time.monotonic()
        if until > now:
            # маленький jitter, чтобы воркеры не просыпались одновременно
            await asyncio.sleep((until - now) + random.uniform(0.0, 0.3))

    async def mark_done(idx: int, ok: bool, unchanged: bool, source: str) -> None:
        async with lock:
            st.processed += 1

            if ok:
                st.ok += 1
                if unchanged:
                    st.unchanged += 1
                else:
                    st.by_source[source] = int(st.by_source.get(source, 0)) + 1

                if hasattr(st, "retry_idx") and idx in st.retry_idx:
                    st.retry_idx.remove(idx)

            else:
                st.failed += 1
                if hasattr(st, "retry_idx"):
                    if idx not in st.retry_idx:
                        st.retry_idx.append(idx)

            # advance contiguous next_idx ONLY for idx >= next_idx
            if idx >= st.next_idx:
                completed.add(idx)
                while st.next_idx in completed:
                    completed.remove(st.next_idx)
                    st.next_idx += 1

            if st.processed % progress_every == 0:
                print(
                    f"PROGRESS: processed={st.processed} ok={st.ok} unchanged={st.unchanged} "
                    f"failed={st.failed} next_idx={st.next_idx} retry={len(getattr(st,'retry_idx',[]))}",
                    flush=True,
                )

            if cfg.logic.resume and (st.processed % flush_every == 0):
                st.saved_at_unix = int(time.time())
                save_state(cfg.logic.checkpoint_file, st)

    # ---- work queue ----
    q: asyncio.Queue[Tuple[int, Any]] = asyncio.Queue()

    seen: Set[int] = set()

    # сначала ретраи старых фейлов (idx < next_idx)
    for i in list(getattr(st, "retry_idx", [])):
        if 0 <= i < total and i < st.next_idx and i not in seen:
            q.put_nowait((i, seeds[i]))
            seen.add(i)

    # потом основной поток
    for i in range(st.next_idx, total):
        if i not in seen:
            q.put_nowait((i, seeds[i]))
            seen.add(i)

    # ---- workers ----
    async def worker_loop(worker_id: int, session: aiohttp.ClientSession):
        while True:
            try:
                idx, seed = await q.get()
            except asyncio.CancelledError:
                raise

            try:
                if stop_event.is_set():
                    return

                # host cooldown check (for 429)
                host = ""
                try:
                    host = (urlsplit(seed.url).hostname or "").lower()
                except Exception:
                    host = ""

                await wait_host_if_cooled(host)

                ok, unchanged = await process_one(cfg, col, limiter, session, seed)
                await mark_done(idx, ok=ok, unchanged=unchanged, source=seed.source)

            except RetryableHttpError as e:
                # 429 (или другой retryable), не считаем как failed и НЕ делаем mark_done
                retry429_count[idx] = retry429_count.get(idx, 0) + 1

                # включаем cooldown на хост
                await set_host_cooldown(e.host, e.retry_after_sec)

                logger.warning(
                    f"RETRYABLE: idx={idx} source={seed.source} status={e.status} host={e.host} "
                    f"wait={e.retry_after_sec:.2f}s tries={retry429_count[idx]}/{max_429_retries} url={e.url}"
                )
                print(
                    f"RETRYABLE: idx={idx} source={seed.source} status={e.status} wait={e.retry_after_sec:.2f}s url={e.url}",
                    flush=True,
                )

                if stop_event.is_set():
                    return

                if retry429_count[idx] <= max_429_retries:
                    # ждём и ставим обратно в очередь (без mark_done)
                    await asyncio.sleep(e.retry_after_sec)
                    await q.put((idx, seed))
                else:
                    # слишком много 429 по одному idx: считаем как failed (иначе вечный цикл)
                    logger.error(
                        f"GIVEUP_429: idx={idx} source={seed.source} host={e.host} url={e.url} tries={retry429_count[idx]}"
                    )
                    print(
                        f"GIVEUP_429: idx={idx} source={seed.source} url={e.url}",
                        file=sys.stderr,
                        flush=True,
                    )
                    await mark_done(idx, ok=False, unchanged=False, source=seed.source)

            except asyncio.CancelledError:
                raise

            except Exception as e:
                logger.error(f"FAIL: idx={idx} source={seed.source} url={seed.url} err={e}")
                logger.error(traceback.format_exc())
                print(f"FAIL: idx={idx} source={seed.source} url={seed.url} err={e}", file=sys.stderr, flush=True)
                await mark_done(idx, ok=False, unchanged=False, source=seed.source)

            finally:
                q.task_done()

    # aiohttp session
    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        workers = [asyncio.create_task(worker_loop(i, session)) for i in range(int(cfg.logic.concurrency))]

        join_task = asyncio.create_task(q.join())
        stop_task = asyncio.create_task(stop_event.wait())

        done, pending = await asyncio.wait({join_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)

        if stop_task in done and not join_task.done():
            print("Stopping: cancelling workers and draining queue...", flush=True)

            for t in workers:
                t.cancel()

            while True:
                try:
                    _ = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                else:
                    q.task_done()

            await join_task

        for t in workers:
            t.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    # final checkpoint
    if cfg.logic.resume:
        st.saved_at_unix = int(time.time())
        save_state(cfg.logic.checkpoint_file, st)

    # summary
    print("CRAWL SUMMARY:", flush=True)
    print(f"  seeds_total: {total}", flush=True)
    print(f"  processed:   {st.processed}", flush=True)
    print(f"  ok:          {st.ok}", flush=True)
    print(f"  unchanged:   {st.unchanged}", flush=True)
    print(f"  failed:      {st.failed}", flush=True)
    print(f"  by_source_new_docs: {st.by_source}", flush=True)
    print(f"  next_idx:    {st.next_idx}", flush=True)

    listener.stop()
    return 0



def main() -> int:

    if len(sys.argv) >= 2:
        cfg_path = sys.argv[1]
    else:
        cfg_path = os.getenv("CONFIG_PATH", "/config/config.yaml")

    cfg = parse_cfg(load_config(cfg_path))

    if cfg.logic.concurrency < 1 or cfg.logic.per_host_concurrency < 1:
        raise RuntimeError("logic.concurrency and logic.per_host_concurrency must be >= 1")

    if cfg.logic.per_host_concurrency > cfg.logic.concurrency:
        pass

    return asyncio.run(run_crawler(cfg))


if __name__ == "__main__":
    raise SystemExit(main())
