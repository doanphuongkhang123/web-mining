"""Collect the compact evidence used to match SNAP records to English Wikipedia RfAs."""

import argparse
from collections import defaultdict
from datetime import datetime
from email.utils import parsedate_to_datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .preprocess import DEFAULT_EVIDENCE, DATE_PATTERN, MONTH_LOOKUP, STAMP_FORMAT, load_snap, normalized_name
from .source import ensure_dataset, sha256_file

API = "https://en.wikipedia.org/w/api.php"
ARCHIVE_PREFIX = "Wikipedia:Requests for adminship by year/"
PAGE_PREFIX = "Wikipedia:Requests for adminship/"
ARCHIVE_ROW = re.compile(r"^\s*\{\{Recent RfX\|A\|([^|]+)\|([^|]*)\|([^|]+)\|\d+\|\d+\|\d+\|([^|}]+)\}\}", re.I | re.M)
SIGNATURE = re.compile(DATE_PATTERN + r"\s*\(UTC\)", re.I)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
CLOSURE = re.compile(r"\b(?:clos(?:ed|ing|ure)|withdrawn|consensus|promoted|final)\b", re.I)
LAST_REQUEST = 0.0
REQUEST_INTERVAL = 6.2


def request_api(**params):
    global LAST_REQUEST
    query = urlencode({"action": "query", "format": "json", "formatversion": 2, "maxlag": 5, **params})
    request = Request(f"{API}?{query}", headers={
        "User-Agent": "VoteLensCourseProject/1.0 (read-only historical RfA metadata)",
        "Accept-Encoding": "gzip",
    })
    for attempt in range(5):
        delay = 2 ** attempt
        time.sleep(max(0.0, LAST_REQUEST + REQUEST_INTERVAL - time.monotonic()))
        LAST_REQUEST = time.monotonic()
        try:
            with urlopen(request, timeout=14) as response:
                body = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
            data = json.loads(body)
            if "error" not in data:
                return data["query"]
            if data["error"]["code"] != "maxlag":
                raise ValueError(data["error"])
            delay = max(5.0, delay)
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
            retry_after = error.headers.get("Retry-After", "")
            if retry_after.isdigit():
                delay = max(delay, int(retry_after))
            elif retry_after:
                retry_time = parsedate_to_datetime(retry_after)
                delay = max(delay, (retry_time - datetime.now(retry_time.tzinfo)).total_seconds())
            elif error.code == 429:
                delay = 60
            print(f"Wikipedia returned HTTP {error.code}. Retrying in {delay:.0f}s.", flush=True)
        except (URLError, TimeoutError, socket.timeout):
            if attempt == 4:
                raise
        time.sleep(delay)
    raise RuntimeError("Wikipedia API stayed above its maxlag threshold")


def get_pages(titles):
    # POST-free requests are batched to stay within the MediaWiki API limits.
    params = {
        "titles": "|".join(titles), "redirects": 1, "prop": "revisions",
        "rvprop": "ids|timestamp|content", "rvslots": "main",
    }
    return request_api(**params)


def revision(page):
    return next(iter(page.get("revisions", [])), {})


def wikitext(page):
    return revision(page).get("slots", {}).get("main", {}).get("content", "")


def signature_time(match):
    try:
        return datetime(
            int(match["year"]), MONTH_LOOKUP[match["month"].lower()], int(match["day"]),
            int(match["hour"]), int(match["minute"]),
        ).strftime(STAMP_FORMAT)
    except (KeyError, ValueError):
        return ""


def page_evidence(page):
    text = HTML_COMMENT.sub("", wikitext(page))
    if not text:
        return set(), []
    signatures = {signature_time(match) for match in SIGNATURE.finditer(text)} - {""}
    first_section = re.search(r"^={3,}\s*(?:nomination|questions?|support|discussion|candidate)", text, re.I | re.M)
    header = text[:first_section.start()] if first_section else text[:2000]
    closings = {
        signature_time(match)
        for line in header.splitlines() if CLOSURE.search(line)
        for match in SIGNATURE.finditer(line)
    } - {""}
    return signatures, sorted(closings)


def collect(data_path, target, cache):
    raw = load_snap(data_path)
    data_sha = sha256_file(data_path)
    relevant = set(raw.candidate_id)
    candidates = defaultdict(list)
    for number, group in raw.groupby("block_id", sort=False):
        starts = group.timestamp.dropna().dt.strftime(STAMP_FORMAT)
        candidates[group.candidate_id.iat[0]].append({
            "block_id": int(number), "times": dict(zip(group.loc[starts.index, "record_id"].map(int), starts)),
        })

    index = get_pages([f"{ARCHIVE_PREFIX}{year}" for year in range(2004, 2015)])
    entries = defaultdict(list)
    index_revisions = {}
    for page in index["pages"]:
        index_revisions[page["title"]] = revision(page)["revid"]
        for match in ARCHIVE_ROW.finditer(wikitext(page)):
            candidate, attempt, date, outcome = (value.strip() for value in match.groups())
            key = normalized_name(candidate)
            if key not in relevant:
                continue
            listed_date = pd.to_datetime(date, format="%d %B %Y", errors="coerce")
            if pd.isna(listed_date):
                continue
            title = f"{PAGE_PREFIX}{candidate}{' ' + attempt if attempt else ''}"
            entries[title].append({
                "candidate_id": key, "archive_date": listed_date.strftime("%Y-%m-%d"),
                "archive_outcome": 1 if outcome == "S" else -1,
                "index_revision_id": revision(page)["revid"],
            })

    titles = sorted(entries)
    pages = []
    batch_size = 45
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(titles), batch_size):
        batch = titles[offset:offset + batch_size]
        cache_key = hashlib.sha256(
            repr((data_sha, index_revisions, batch)).encode()
        ).hexdigest()
        cache_file = cache / f"{cache_key}.json" if cache is not None else None
        if cache_file is not None and cache_file.exists():
            pages.extend(json.loads(cache_file.read_text(encoding="utf-8")))
            continue
        response = get_pages(batch)
        title_map = {item["from"]: item["to"] for kind in ("normalized", "redirects") for item in response.get(kind, [])}
        resolved = {page["title"]: page for page in response["pages"]}
        batch_evidence = []
        for original in batch:
            title = title_map.get(original, original)
            page = resolved.get(title_map.get(title, title), {})
            if not revision(page):
                continue
            signatures, close_candidates = page_evidence(page)
            for entry in entries[original]:
                matches = []
                for block in candidates[entry["candidate_id"]]:
                    record_ids = [i for i, stamp in block["times"].items() if stamp in signatures]
                    distinct_times = {block["times"][i] for i in record_ids}
                    if distinct_times:
                        matches.append({"block_id": block["block_id"], "record_ids": record_ids,
                                        "distinct_times": len(distinct_times)})
                if matches:
                    batch_evidence.append({
                        **entry, "title": page["title"], "page_id": page["pageid"],
                        "revision_id": revision(page)["revid"], "revision_at": revision(page)["timestamp"],
                        "close_candidates": close_candidates, "matches": matches,
                    })
        pages.extend(batch_evidence)
        if cache_file is not None:
            temporary = cache_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(batch_evidence, ensure_ascii=False), encoding="utf-8")
            temporary.replace(cache_file)
        if offset % (batch_size * 5) == 0:
            print(f"Requested {min(offset + batch_size, len(titles))}/{len(titles)} archive pages", flush=True)
    payload = {
        "version": 1, "source_sha256": data_sha,
        "index_revisions": index_revisions, "titles_requested": len(titles), "pages": pages,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(gzip.compress(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(), mtime=0))
    temporary.replace(target)
    print(f"Saved {len(pages)} archive evidence entries to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Local SNAP gzip file")
    parser.add_argument("--output", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--cache", type=Path, help="Optional cache for resuming interrupted downloads")
    args = parser.parse_args()
    collect(args.data or ensure_dataset(), args.output, args.cache)
