"""Load SNAP vote records and locate the local toxicity scores."""

import gzip
import hashlib
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "rfa_toxicity"
DEFAULT_SNAP = DATA_DIR / "wiki-RfA.txt.gz"
DEFAULT_SCORED = DATA_DIR / "wikipedia-rfa-toxicity.zip"

SNAP_DOWNLOAD = "https://snap.stanford.edu/data/wiki-RfA.txt.gz"
SNAP_SHA256 = "88d53196fb2564a2e20286dbba818832f718cc352bb181a2101d23d2556f0862"
FIELDS = ("SRC", "TGT", "VOT", "RES", "YEA", "DAT", "TXT")


def snap_path(path=None):
    """Use a local SNAP file or download and verify the default copy."""
    if path is not None:
        return Path(path)
    path = DEFAULT_SNAP
    if path.exists():
        with path.open("rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
        if checksum != SNAP_SHA256:
            raise ValueError(f"Unexpected checksum for {path}. Pass a SNAP file with --data.")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(SNAP_DOWNLOAD, headers={"User-Agent": "VoteLens Web Mining coursework/1.0"})
    with TemporaryDirectory(prefix=".download-", dir=path.parent) as temp:
        staged = Path(temp) / path.name
        with urlopen(request, timeout=60) as response, staged.open("wb") as target:
            shutil.copyfileobj(response, target)
        with staged.open("rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
        if checksum != SNAP_SHA256:
            raise ValueError("Unexpected checksum for the SNAP download")
        staged.replace(path)
    return path


def scored_path(path=None):
    path = Path(path) if path is not None else DEFAULT_SCORED
    if not path.is_file():
        raise FileNotFoundError(f"Missing toxicity scores at {path}. Pass a local ZIP or CSV with --scored.")
    return path


def read_snap(path):
    """Read the blank-line-delimited SNAP records and keep colons inside values."""
    gzip_input = Path(path).suffix == ".gz"
    opener = gzip.open if gzip_input else open
    records = []
    current = {}
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            row = line.rstrip("\r\n")
            if not row:
                if current:
                    records.append(current)
                    current = {}
                continue
            field, separator, value = row.partition(":")
            if not separator or field not in FIELDS or field in current:
                raise ValueError(f"Invalid SNAP field at line {number}")
            current[field] = value
    if current:
        records.append(current)

    votes = pd.DataFrame.from_records(records, columns=FIELDS)
    if votes.empty or votes.isna().any().any():
        raise ValueError("SNAP input needs a complete set of seven fields per record")
    return votes.rename(columns=str.lower)
