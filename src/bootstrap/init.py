#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import os
import sys
import time
import shutil
import subprocess
import pathlib
import random
import time
import re
from dataclasses import dataclass
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Callable, Iterable, Set, Tuple
from urllib.parse import quote

import yaml
import pymysql
import requests

_ALLOWED_RESOURCE = re.compile(r"^[a-z0-9_]+$")

def getenv(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Missing env var: {name}")
    return v



def load_config() -> Dict[str, Any]:
    path = os.getenv("CONFIG_PATH", "/config/config.yaml")
    p = pathlib.Path(path)
    if not p.exists():
        raise RuntimeError(f"Config file not found at CONFIG_PATH: {p}")

    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise RuntimeError(f"Config is empty YAML: {p}")

    if not isinstance(cfg, dict):
        raise RuntimeError(f"YAML root must be an object/dict: {p}")

    return cfg

def require_obj(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key not in d:
        raise RuntimeError(f"Missing config section: {key}")
    v = d[key]
    if not isinstance(v, dict):
        raise RuntimeError(f"Config section '{key}' must be an object")
    return v


def require(d: Dict[str, Any], key: str, expected: Any) -> Any:
    if key not in d:
        raise RuntimeError(f"Missing config.bootstrap.{key}")
    v = d[key]
    if expected is not None and not isinstance(v, expected):
        raise RuntimeError(f"config.bootstrap.{key} must be {expected}, got {type(v)}")
    return v




class StepFailed(RuntimeError):
    def __init__(self, step: str, original: BaseException):
        super().__init__(f"Step '{step}' failed: {original}")
        self.step = step
        self.original = original

def run_step(step_name: str, fn: Callable[[], Any]) -> Any:
    try:
        print(f"==> {step_name}")
        return fn()
    except Exception as e:
        raise StepFailed(step_name, e) from e



@dataclass(frozen=True)
class BootstrapConfig:
    wiki_projects: List[str]
    dump_resources: List[str]
    download_dumps: bool
    import_dumps: bool
    export_seeds: bool
    seed_category: str
    max_depth: int
    seed_limit: int
    bd_attempts_count: int
    bd_attempts_dly: float

    timeout: Tuple[float, float]

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> "BootstrapConfig":
        b = require_obj(cfg, "bootstrap")

        projects = require(b, "wiki_projects", list)
        if not all(isinstance(x, str) and x.strip() for x in projects):
            raise RuntimeError("config.bootstrap.wiki_projects must be a non-empty list of strings")
        projects = [x.strip() for x in projects]

        t = require(b, "timeout_sec", dict)
        connect = float(require(t, "connect", (int, float)))
        read = float(require(t, "read", (int, float)))

        return BootstrapConfig(
            wiki_projects=projects,
            dump_resources=require(b, "dump_resources", list),
            download_dumps=require(b, "download_dumps", bool),
            import_dumps=require(b, "import_dumps", bool),
            export_seeds=require(b, "export_seeds", bool),
            seed_category=require(b, "seed_category", str),
            max_depth=require(b, "max_depth", int),
            seed_limit=require(b, "seed_limit", int),
            bd_attempts_count=require(b, "bd_attempts_count", int),
            bd_attempts_dly=float(require(b, "bd_attempts_dly", (int, float))),

            timeout=(connect, read)
        )


@dataclass(frozen=True)
class MySQLConnInfo:
    host: str
    port: int
    user: str
    password: str

def mysql_info() -> MySQLConnInfo:
    return MySQLConnInfo(
        host=os.getenv("MYSQL_HOST", "mysql"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
    )


def mysql_connect(db: Optional[str] = None):
    info = mysql_info()
    return pymysql.connect(
        host=info.host,
        port=info.port,
        user=info.user,
        password=info.password,
        database=db,
        charset="utf8mb4",
        autocommit=True,
    )

def mysql_cli_cmd(db: str) -> list[str]:
    info = mysql_info()
    return [
        "mysql",
        "-h", info.host,
        "-P", str(info.port),
        "-u", info.user,
        "--protocol=tcp",
        "--default-character-set=utf8mb4",
        "--local-infile=1",
        db,
    ]

def check_connection(attempts: int, dly_sec: float) -> bool:
    last_err: Optional[Exception] = None
    
    for attempt in range(1, attempts+1):
        try:
            conn = mysql_connect()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    _ = cur.fetchone()
            finally:
                conn.close()

            print("OK: MySQL is up and accepts connections.")
            return True
        
        except Exception as e:
            last_err = e
            print(f"WARN: MySQL not ready (attempt {attempt}/{attempts}): {e}", file=sys.stderr)
            time.sleep(dly_sec)

    raise RuntimeError(f"MySQL not ready after {attempts} attempts. Last error: {last_err}")
    


def create_databases(projects: List[str]) -> bool:

    conn = mysql_connect()
    try:
        with conn.cursor() as cur:
            for db in projects:
                cur.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{db}` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
                )
        with conn.cursor() as cur:
            cur.execute("SHOW DATABASES;")
            dbs = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()

    missing = [db for db in projects if db not in dbs]
    if missing:
        raise RuntimeError(f"Databases not created: {missing}")

    print(f"OK: databases exist: {projects}")
    return True

def dump_filename(project: str, resource: str) -> str:
    r = resource.strip()
    if not r:
        raise RuntimeError("Empty dump resource")
    if not _ALLOWED_RESOURCE.match(r):
        raise RuntimeError(f"Bad dump resource name: '{resource}'")
    return f"{project}-latest-{r}.sql.gz"


def download_dumps(
    boot: Any,
    dumps_dir: str | pathlib.Path,
    *,
    overwrite: bool = False,
    timeout: tuple[float, float],  # (connect_timeout, read_timeout)
    progress_every_mb: int = 50,
) -> Dict[str, Dict[str, str]]:


    projects: Iterable[str] = getattr(boot, "wiki_projects", None)
    if not projects:
        raise RuntimeError("boot.wiki_projects is missing or empty")

    resources: Iterable[str] = getattr(boot, "dump_resources", None)
    if not resources:
        raise RuntimeError("boot.dump_resources is missing or empty")

    dumps_dir = pathlib.Path(dumps_dir)


    result: Dict[str, Dict[str, str]] = {}

    for proj_raw in projects:
        proj = str(proj_raw).strip()
        if not proj:
            continue

        result[proj] = {}
        base = f"https://dumps.wikimedia.org/{proj}/latest"

        for res_raw in resources:
            res = str(res_raw).strip()

            filename = dump_filename(proj, res)
            url = f"{base}/{filename}"

            dest = dumps_dir / proj / filename
            dest.parent.mkdir(parents=True, exist_ok=True)

            if dest.exists() and dest.stat().st_size > 0 and not overwrite:
                print(f"SKIP: '{filename}' already exists ({dest.stat().st_size} bytes)")
                result[proj][res] = "skipped"
                continue

            tmp = dest.with_suffix(dest.suffix + ".part")

            print(f"Downloading of file '{filename}' was started.")
            print(f"  URL:  {url}")
            print(f"  Dest: {dest}")

            start_ts = time.time()
            bytes_written = 0
            next_progress = progress_every_mb * 1024 * 1024

            try:
                with requests.get(url, stream=True, timeout=timeout) as r:
                    print(f"  HTTP: {r.status_code} {r.reason}")
                    r.raise_for_status()

                    total = r.headers.get("Content-Length")
                    total_int: Optional[int] = int(total) if total and total.isdigit() else None
                    if total_int:
                        print(f"  Size: {total_int / (1024*1024):.1f} MB")

                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            f.write(chunk)
                            bytes_written += len(chunk)

                            if bytes_written >= next_progress:
                                mb = bytes_written / (1024 * 1024)
                                if total_int:
                                    pct = (bytes_written / total_int) * 100
                                    print(f"  Progress: {mb:.1f} MB ({pct:.1f}%)")
                                else:
                                    print(f"  Progress: {mb:.1f} MB")
                                next_progress += progress_every_mb * 1024 * 1024

                tmp.replace(dest)

                elapsed = time.time() - start_ts
                mb = bytes_written / (1024 * 1024)
                speed = (mb / elapsed) if elapsed > 0 else 0.0

                print(f"Downloading of file '{filename}' was finished successfully.")
                print(f"  Downloaded: {mb:.1f} MB in {elapsed:.1f}s ({speed:.1f} MB/s)")

                result[proj][res] = "downloaded"

            except Exception as e:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass

                print(f"Downloading of file '{filename}' was finished unsuccessfully.", file=sys.stderr)
                print(f"  Error: {e}", file=sys.stderr)
                result[proj][res] = f"failed: {e}"

    return result



def _dump_signature(path: pathlib.Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "size": st.st_size,
        "mtime": int(st.st_mtime),
    }


def _marker_path(logs_dir: pathlib.Path, project: str, resource: str) -> pathlib.Path:
    return logs_dir / "import_markers" / f"{project}__{resource}.json"


def _already_imported(marker_file: pathlib.Path, sig: Dict[str, Any]) -> bool:
    if not marker_file.exists():
        return False
    try:
        data = json.loads(marker_file.read_text(encoding="utf-8"))
        return data.get("size") == sig["size"] and data.get("mtime") == sig["mtime"]
    except Exception:
        return False


def _write_marker(marker_file: pathlib.Path, sig: Dict[str, Any], extra: Dict[str, Any]) -> None:
    marker_file.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(sig)
    payload.update(extra)
    marker_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")



def import_dumps(
    boot: Any,
    dumps_dir: str | pathlib.Path,
    *,
    logs_dir: str | pathlib.Path = "/data/logs",
    overwrite: bool = False,
) -> Dict[str, Dict[str, str]]:

    projects: Iterable[str] = getattr(boot, "wiki_projects", None)
    if not projects:
        raise RuntimeError("boot.wiki_projects is missing or empty")

    resources: Iterable[str] = getattr(boot, "dump_resources", None)
    if not resources:
        raise RuntimeError("boot.dump_resources is missing or empty")

    dumps_dir = pathlib.Path(dumps_dir)
    logs_dir = pathlib.Path(logs_dir)


    pwd = mysql_info().password
    if pwd is None or str(pwd).strip() == "":
        raise RuntimeError(
            "MYSQL_PASSWORD is empty inside bootstrap container. "
            "Check .env loading and docker-compose environment for bootstrap."
        )
    
    env = os.environ.copy()
    env["MYSQL_PWD"] = pwd

    result: Dict[str, Dict[str, str]] = {}

    for proj_raw in projects:
        proj = str(proj_raw).strip()
        if not proj:
            continue

        result[proj] = {}

        for res_raw in resources:
            res = str(res_raw).strip()

            filename = dump_filename(proj, res)
            dump_path = dumps_dir / proj / filename

            if not dump_path.exists() or dump_path.stat().st_size == 0:
                raise RuntimeError(f"Dump file missing or empty: {dump_path}")

            sig = _dump_signature(dump_path)
            marker = _marker_path(logs_dir, proj, res)

            if not overwrite and _already_imported(marker, sig):
                print(f"SKIP: '{filename}' already imported (marker matched).")
                result[proj][res] = "skipped"
                continue

            print(f"Importing of file '{filename}' was started.")
            print(f"  DB:   {proj}")
            print(f"  File: {dump_path}")

            start_ts = time.time()
            mysql_cmd = mysql_cli_cmd(proj)

            try:
                with subprocess.Popen(
                    mysql_cmd,
                    stdin=subprocess.PIPE,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    env=env,
                ) as proc:
                    assert proc.stdin is not None

                    with gzip.open(dump_path, "rb") as gz:
                        shutil.copyfileobj(gz, proc.stdin, length=1024 * 1024)

                    proc.stdin.close()
                    rc = proc.wait()

                if rc != 0:
                    raise RuntimeError(f"mysql exited with code {rc}")

                elapsed = time.time() - start_ts
                print(f"Importing of file '{filename}' was finished successfully.")
                print(f"  Time: {elapsed:.1f}s")

                _write_marker(
                    marker,
                    sig,
                    {
                        "imported_at_unix": int(time.time()),
                        "project": proj,
                        "resource": res,
                        "file": filename,
                    },
                )

                result[proj][res] = "imported"

            except Exception as e:
                print(f"Importing of file '{filename}' was finished unsuccessfully.", file=sys.stderr)
                print(f"  Error: {e}", file=sys.stderr)
                result[proj][res] = f"failed: {e}"

    failed = []
    for proj, items in result.items():
        for res, status in items.items():
            if isinstance(status, str) and status.startswith("failed:"):
                failed.append((proj, res, status))

    if failed:
        details = "\n".join([f"- {p}/{r}: {s}" for p, r, s in failed])
        raise RuntimeError(f"Import failed for {len(failed)} dump(s):\n{details}")

    return result




def check_tables(
    boot: "BootstrapConfig",
    *,
    only_tables: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:

    errors: List[Tuple[str, str]] = []
    summary: Dict[str, Dict[str, Any]] = {}

    if only_tables is not None:
        expected_tables = [t.strip() for t in only_tables if t and t.strip()]
    else:
        expected_tables = [t.strip() for t in boot.dump_resources if t and str(t).strip()]

    if not expected_tables:
        raise RuntimeError("No tables selected for checking (expected_tables is empty)")

    for db_raw in boot.wiki_projects:
        db = str(db_raw).strip()
        if not db:
            continue

        try:
            conn = mysql_connect(db)
            try:
                with conn.cursor() as cur:

                    cur.execute("SHOW TABLES;")
                    tables_all = [row[0] for row in cur.fetchall()]
                    tables_set = set(tables_all)

                    print(f"DB '{db}' tables ({len(tables_all)}):")
                    for t in tables_all:
                        print(f"  - {t}")

                    missing = [t for t in expected_tables if t not in tables_set]
                    if missing:
                        raise RuntimeError(f"Missing expected tables: {missing}")

                    table_info: Dict[str, Any] = {}

                    for table in expected_tables:
                        info: Dict[str, Any] = {}

                        cur.execute(
                            """
                            SELECT table_rows, data_length, index_length
                            FROM information_schema.tables
                            WHERE table_schema=%s AND table_name=%s
                            """,
                            (db, table),
                        )
                        meta = cur.fetchone()
                        if meta:
                            rows_est, data_len, idx_len = meta
                            info["rows_est"] = int(rows_est) if rows_est is not None else None
                            info["data_mb"] = (int(data_len) / (1024 * 1024)) if data_len is not None else None
                            info["index_mb"] = (int(idx_len) / (1024 * 1024)) if idx_len is not None else None
                        else:
                            info["rows_est"] = None
                            info["data_mb"] = None
                            info["index_mb"] = None

                        cur.execute(
                            """
                            SELECT ordinal_position, column_name, column_type
                            FROM information_schema.columns
                            WHERE table_schema=%s AND table_name=%s
                            ORDER BY ordinal_position;
                            """,
                            (db, table),
                        )
                        rows = cur.fetchall()
                        cols = [{"pos": int(pos), "name": name, "type": ctype} for (pos, name, ctype) in rows]
                        info["columns"] = cols

                        table_info[table] = info

                        print(
                            f"INFO: {db}.{table} rows_est={info.get('rows_est')} "
                            f"data_mb={info.get('data_mb')} index_mb={info.get('index_mb')}"
                        )
                        print(f"  Columns ({len(cols)}):")
                        for c in cols:
                            print(f"    - {c['pos']:>2}. {c['name']} | {c['type']}")

                    summary[db] = {
                        "tables_all": tables_all,
                        "tables_selected": expected_tables,
                        "table_info": table_info,
                    }

            finally:
                conn.close()

        except Exception as e:
            msg = str(e)
            errors.append((db, msg))
            print(f"FAIL: check_tables for DB '{db}': {msg}", file=sys.stderr)

    if errors:
        details = "\n".join([f"- {db}: {msg}" for db, msg in errors])
        raise RuntimeError(f"check_tables failed for {len(errors)} database(s):\n{details}")

    return summary




def _decode_bytes(v: Any) -> Any:
    """bytes -> utf-8 (если можно), иначе короткий hex-превью."""
    if isinstance(v, (bytes, bytearray)):
        try:
            s = v.decode("utf-8", errors="strict")

            if any(ord(ch) < 9 for ch in s):
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "control chars")
            return s
        except Exception:
            hx = v.hex()
            preview = hx[:64] + ("..." if len(hx) > 64 else "")
            return f"0x{preview} (len={len(v)})"
    return v


def _row_as_dict(cols: List[str], row: Tuple[Any, ...]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for i, name in enumerate(cols):
        out[name] = _decode_bytes(row[i]) if i < len(row) else None
    return out




def print_samples(
    boot: "BootstrapConfig",
    schema: Dict[str, Dict[str, Any]],
    *,
    sample_rows: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    Печатает sample_rows строк из таблиц schema[db]["tables_selected"].
    Возвращает samples_summary (можно потом сохранить в json).
    """
    errors: List[Tuple[str, str]] = []
    samples_summary: Dict[str, Dict[str, Any]] = {}

    for db, db_info in schema.items():
        try:
            tables_selected = db_info.get("tables_selected", [])
            table_info = db_info.get("table_info", {}) or {}

            if not tables_selected:
                raise RuntimeError(f"schema['{db}'].tables_selected is empty")

            conn = mysql_connect(db)
            try:
                with conn.cursor() as cur:
                    print(f"DB '{db}' samples (LIMIT {sample_rows}):")
                    samples_summary[db] = {}

                    for table in tables_selected:

                        cols_meta = (table_info.get(table, {}) or {}).get("columns", [])
                        all_cols = [c["name"] for c in cols_meta if isinstance(c, dict) and "name" in c]

                        if not all_cols:
                            raise RuntimeError(f"No columns found in schema for {db}.{table}")

                        if table == "page":
                            preferred = [
                                "page_id",
                                "page_namespace",
                                "page_title",
                                "page_is_redirect",
                                "page_len",
                                "page_latest",
                                "page_content_model",
                            ]
                            select_cols = [c for c in preferred if c in all_cols]
                            if not select_cols:
                                select_cols = all_cols[:6]

                        elif table == "categorylinks":
                            preferred = [
                                "cl_from",
                                "cl_type",
                                "cl_timestamp",
                                "cl_target_id",
                                "cl_collation_id",
                            ]
                            select_cols = [c for c in preferred if c in all_cols]
                            if not select_cols:
                                blacklist = {"cl_sortkey", "cl_sortkey_prefix"}
                                select_cols = [c for c in all_cols if c not in blacklist][:6]

                        else:
                            select_cols = all_cols[:6]

                        cols_sql = ", ".join([f"`{c}`" for c in select_cols])
                        q = f"SELECT {cols_sql} FROM `{table}` LIMIT %s;"
                        cur.execute(q, (int(sample_rows),))
                        rows = cur.fetchall() or []

                        print(f"  Table '{table}': selected_cols={select_cols}, rows={len(rows)}")

                        pretty_rows = []
                        for idx, r in enumerate(rows, 1):
                            d = _row_as_dict(select_cols, r)
                            pretty_rows.append(d)
                            print(f"    sample[{idx}]: {d}")

                        samples_summary[db][table] = {
                            "selected_cols": select_cols,
                            "rows": pretty_rows,
                        }

            finally:
                conn.close()

        except Exception as e:
            msg = str(e)
            errors.append((db, msg))
            print(f"FAIL: print_samples for DB '{db}': {msg}", file=sys.stderr)

    if errors:
        details = "\n".join([f"- {db}: {msg}" for db, msg in errors])
        raise RuntimeError(f"print_samples failed for {len(errors)} database(s):\n{details}")

    return samples_summary





def _decode_title(v: Any) -> str:
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")
    return str(v)


def _wiki_title_norm(title: str) -> str:
    return title.replace(" ", "_")


def _make_url(domain: str, title: str) -> str:
    title = _wiki_title_norm(title)
    return f"https://{domain}/wiki/{quote(title, safe='()_:-')}"


def _sources_from_cfg(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources = cfg.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("config.sources must be a non-empty list")

    out = []
    for i, s in enumerate(sources):
        if not isinstance(s, dict):
            raise RuntimeError(f"config.sources[{i}] must be an object")
        name = str(s.get("name", "")).strip()
        if not name:
            raise RuntimeError(f"config.sources[{i}].name is required")

        allowed = s.get("allowed_domains")
        if not isinstance(allowed, list) or not allowed or not str(allowed[0]).strip():
            raise RuntimeError(f"config.sources[{i}].allowed_domains must have at least 1 domain")
        domain = str(allowed[0]).strip()

        out.append({"name": name, "domain": domain})
    return out


def _collect_seeds_for_project(
    boot: Any,
    proj: str,
    domain: str,
    *,
    limit: int,
    only_main_namespace: bool,
    exclude_redirects: bool,
) -> List[Tuple[str, str, int, int, str]]:
    seed_category = getattr(boot, "seed_category", None)
    if not seed_category or not str(seed_category).strip():
        raise RuntimeError("boot.seed_category is missing or empty")

    max_depth = int(getattr(boot, "max_depth", 1))
    limit = int(limit)
    if limit <= 0:
        return []

    conn = mysql_connect(proj)
    try:
        with conn.cursor() as cur:
            cat_title = _wiki_title_norm(str(seed_category).strip())

            cur.execute(
                """
                SELECT lt_id
                FROM linktarget
                WHERE lt_namespace=14 AND lt_title=%s
                LIMIT 1;
                """,
                (cat_title,),
            )
            row = cur.fetchone()
            if not row:
                print(f"WARN: [{proj}] seed category not found: Category:{cat_title}")
                return []

            start_lt_id = int(row[0])


            q: Deque[Tuple[int, int]] = deque([(start_lt_id, 0)])
            visited_cats: Set[int] = {start_lt_id}
            picked_pages: Set[int] = set()

            out: List[Tuple[str, str, int, int, str]] = []

            while q and len(out) < limit:
                cat_lt_id, depth = q.popleft()
                if depth > max_depth:
                    continue

                cur.execute(
                    """
                    SELECT cl_from, cl_type
                    FROM categorylinks
                    WHERE cl_target_id=%s;
                    """,
                    (cat_lt_id,),
                )
                members = cur.fetchall() or []
                if not members:
                    continue

                page_ids: List[int] = []
                subcat_ids: List[int] = []

                for cl_from, cl_type in members:
                    pid = int(cl_from)

                    # cl_type бывает bytes (b'page'), декодим
                    t = _decode_title(cl_type).strip().lower()

                    if t == "page":
                        page_ids.append(pid)
                    elif t == "subcat":
                        subcat_ids.append(pid)

                if page_ids and len(out) < limit:
                    CHUNK = 1000
                    for i in range(0, len(page_ids), CHUNK):
                        chunk = page_ids[i:i + CHUNK]
                        if not chunk or len(out) >= limit:
                            break

                        placeholders = ",".join(["%s"] * len(chunk))
                        cur.execute(
                            f"""
                            SELECT page_id, page_namespace, page_title, page_is_redirect
                            FROM page
                            WHERE page_id IN ({placeholders});
                            """,
                            tuple(chunk),
                        )
                        rows = cur.fetchall() or []
                        for page_id, ns, title, is_redir in rows:
                            if len(out) >= limit:
                                break
                            page_id = int(page_id)
                            ns = int(ns)
                            is_redir = int(is_redir)

                            if only_main_namespace and ns != 0:
                                continue
                            if exclude_redirects and is_redir == 1:
                                continue
                            if page_id in picked_pages:
                                continue

                            title_s = _decode_title(title)
                            url = _make_url(domain, title_s)
                            out.append((url, title_s, page_id, ns, proj))
                            picked_pages.add(page_id)

                if subcat_ids and depth < max_depth:
                    CHUNK = 1000
                    for i in range(0, len(subcat_ids), CHUNK):
                        chunk = subcat_ids[i:i + CHUNK]
                        if not chunk:
                            continue

                        placeholders = ",".join(["%s"] * len(chunk))
                        cur.execute(
                            f"""
                            SELECT page_namespace, page_title
                            FROM page
                            WHERE page_id IN ({placeholders});
                            """,
                            tuple(chunk),
                        )
                        cat_pages = cur.fetchall() or []
                        for ns, title in cat_pages:
                            if int(ns) != 14:
                                continue

                            title_s = _wiki_title_norm(_decode_title(title))
                            cur.execute(
                                """
                                SELECT lt_id
                                FROM linktarget
                                WHERE lt_namespace=14 AND lt_title=%s
                                LIMIT 1;
                                """,
                                (title_s,),
                            )
                            r2 = cur.fetchone()
                            if not r2:
                                continue

                            lt_id2 = int(r2[0])
                            if lt_id2 in visited_cats:
                                continue

                            visited_cats.add(lt_id2)
                            q.append((lt_id2, depth + 1))

            return out
    finally:
        conn.close()


def export_seeds(
    boot: Any,
    exports_dir: str | pathlib.Path,
    cfg: Dict[str, Any],
    *,
    out_file: Optional[str] = None,
    only_main_namespace: bool = True,
    exclude_redirects: bool = True,
    rnd_seed: int = 42,
) -> str:
    sources = _sources_from_cfg(cfg)
    if len(sources) < 2:
        raise RuntimeError("Need at least 2 sources in config.sources")

    projects = getattr(boot, "wiki_projects", None)
    if not projects:
        raise RuntimeError("boot.wiki_projects is missing or empty")
    projects = [str(x).strip() for x in projects if str(x).strip()]

    s0 = sources[0]
    s1 = sources[1]

    if s0["name"] not in projects or s1["name"] not in projects:
        raise RuntimeError(
            f"boot.wiki_projects must include the first two config.sources names: "
            f"{s0['name']}, {s1['name']}"
        )

    seeds_cfg = cfg.get("seeds", {}) or {}
    shuffle = bool(seeds_cfg.get("shuffle", False))
    max_per_source = int(seeds_cfg.get("max_per_source", 0))
    if max_per_source <= 0:
        raise RuntimeError("config.seeds.max_per_source must be > 0")

    logic = cfg.get("logic", {}) or {}
    target_total = int(logic.get("max_unique_docs", getattr(boot, "seed_limit", 32000)))
    if target_total <= 0:
        raise RuntimeError("target_total must be > 0")

    exports_dir = pathlib.Path(exports_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)

    cat = _wiki_title_norm(str(getattr(boot, "seed_category", "seed")).strip() or "seed")
    out_path = pathlib.Path(out_file) if out_file else (exports_dir / f"seeds_all_{cat}.tsv")

    cand0 = _collect_seeds_for_project(
        boot, s0["name"], s0["domain"],
        limit=target_total,
        only_main_namespace=only_main_namespace,
        exclude_redirects=exclude_redirects,
    )
    cand1 = _collect_seeds_for_project(
        boot, s1["name"], s1["domain"],
        limit=target_total,
        only_main_namespace=only_main_namespace,
        exclude_redirects=exclude_redirects,
    )

    if shuffle:
        r = random.Random(rnd_seed)
        r.shuffle(cand0)
        r.shuffle(cand1)

    take0_first = min(max_per_source, len(cand0), target_total)
    remaining = target_total - take0_first

    take1 = min(len(cand1), remaining)
    remaining -= take1

    take0_extra = 0
    if remaining > 0 and take0_first < len(cand0):
        take0_extra = min(len(cand0) - take0_first, remaining)
        remaining -= take0_extra

    sel0 = cand0[:take0_first + take0_extra]
    sel1 = cand1[:take1]
    all_rows = sel0 + sel1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("url\ttitle\tpage_id\tnamespace\tsource\n")
        for url, title, page_id, ns, source in all_rows:
            f.write(f"{url}\t{title}\t{page_id}\t{ns}\t{source}\n")

    print("SEEDS RESULT:")
    print(f"  {s0['name']}: {len(sel0)} (first {take0_first} + extra {take0_extra})")
    print(f"  {s1['name']}: {len(sel1)}")
    print(f"  total: {len(all_rows)} / target {target_total}")
    if remaining > 0:
        print(f"WARN: not enough seeds to reach target, missing={remaining}")

    return str(out_path)



def main() -> int:

    try:
        cfg = load_config()
        boot = BootstrapConfig.from_cfg(cfg)

        run_step("Checking mysql connection...", lambda: check_connection(attempts=boot.bd_attempts_count, dly_sec=boot.bd_attempts_dly))
        run_step("Creating databases...", lambda: create_databases(boot.wiki_projects))
        run_step("Downloading dumps...", lambda: download_dumps(boot, dumps_dir="/data/dumps",timeout=boot.timeout))
        run_step("Importing dumps...", lambda: import_dumps(boot, dumps_dir="/data/dumps", logs_dir="/data/logs"))
        schema = run_step("Checking tables...", lambda: check_tables(boot))
        run_step("Sampling rows...", lambda: print_samples(boot, schema, sample_rows=3))
        run_step("Exporting seeds...", lambda: export_seeds(boot, "/data/exports", cfg))
        print("OK: pipeline finished")
        return 0

    except StepFailed as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 2

    except Exception as e:
        print(f"FAIL (unexpected): {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
