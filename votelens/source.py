"""Load the canonical SNAP data and share vote and comment parsing."""

import gzip
import hashlib
import html
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "wiki-RfA.txt.gz"
SOURCE_URL = "https://snap.stanford.edu/data/wiki-RfA.html"
DATA_URL = "https://snap.stanford.edu/data/wiki-RfA.txt.gz"
SNAP_SHA256 = "88d53196fb2564a2e20286dbba818832f718cc352bb181a2101d23d2556f0862"
FIELDS = ["SRC", "TGT", "VOT", "RES", "YEA", "DAT", "TXT"]
TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)


def sha256_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def ensure_dataset(path=DEFAULT_DATA):
    """Download and verify the official SNAP file when the local copy is absent."""
    path = Path(path)
    if path.exists():
        if sha256_file(path) != SNAP_SHA256:
            raise ValueError(f"{path} does not match the expected SNAP checksum")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(DATA_URL, headers={"User-Agent": "VoteLens Web Mining coursework/1.0"})
    with TemporaryDirectory(prefix=".snap-", dir=path.parent) as directory:
        staged = Path(directory) / path.name
        with urlopen(request, timeout=45) as response, staged.open("wb") as output:
            shutil.copyfileobj(response, output)
        if sha256_file(staged) != SNAP_SHA256:
            raise ValueError(f"The download from {DATA_URL} does not match the expected SNAP checksum")
        staged.replace(path)
    return path


def parse_records(path):
    """Read blank-line separated records without losing colons in comments."""
    records, issues, current = [], [], {}
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.rstrip("\r\n")
            if not line:
                if current:
                    records.append(current)
                    current = {}
                continue
            key, separator, value = line.partition(":")
            if not separator or key not in FIELDS or key in current:
                issues.append({"line": line_number, "content": line[:200]})
                continue
            current[key] = value
        if current:
            records.append(current)
    raw = pd.DataFrame(records).reindex(columns=FIELDS)
    return raw, issues


def clean_comment(text):
    """Lightweight, explicit wiki-markup cleaning; not a MediaWiki renderer."""
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"\[\[([^\[\]]+)\]\]", lambda m: m.group(1).split("|")[-1], text)
    text = re.sub(r"\{\{.*?\}\}", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("'''", "").replace("''", "")
    return html.unescape(text)
