"""PMC-VQA data loading + mmap-backed image access (no extraction on disk)."""

from __future__ import annotations

import io
import mmap
import os
import struct
import threading
import zipfile
import zlib

import numpy as np
import pandas as pd
import torch
from PIL import Image

CHOICES = ["A", "B", "C", "D"]

REQUIRED_COLS = ["Figure_path", "Question", "Answer", "Choice A", "Choice B",
                 "Choice C", "Choice D", "Answer_label"]


def load_csv(path: str) -> pd.DataFrame:
    """Load a PMC-VQA CSV and normalize columns."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns in {path}: {missing}")
    df = df.dropna(subset=REQUIRED_COLS).copy()
    df = df[df["Figure_path"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)
    return df


def label_to_idx(label) -> int:
    s = str(label).strip().upper()
    return CHOICES.index(s[0]) if s and s[0] in CHOICES else 0


def options_for(row) -> list[str]:
    return [str(row[f"Choice {c}"]).strip() for c in CHOICES]


def _parse_central_dir(mm: mmap.mmap, zi: int):
    """Parse zip central directory directly from the mmap (no full-file copy).

    Returns dict basename -> (zi, local_header_offset, compress_size, method).
    """
    end = len(mm)
    eocd = mm.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise ValueError("zip EOCD not found")
    count = struct.unpack("<H", mm[eocd + 10:eocd + 12])[0]
    cd_off = struct.unpack("<I", mm[eocd + 16:eocd + 20])[0]
    entries = {}
    pos = cd_off
    for _ in range(count):
        if mm[pos:pos + 4] != b"PK\x01\x02":
            break
        method = struct.unpack("<H", mm[pos + 10:pos + 12])[0]
        csize = struct.unpack("<I", mm[pos + 20:pos + 24])[0]
        fname_len = struct.unpack("<H", mm[pos + 28:pos + 30])[0]
        extra_len = struct.unpack("<H", mm[pos + 30:pos + 32])[0]
        comment_len = struct.unpack("<H", mm[pos + 32:pos + 34])[0]
        local_off = struct.unpack("<I", mm[pos + 42:pos + 46])[0]
        fname = mm[pos + 46:pos + 46 + fname_len].decode("utf-8", "replace")
        entries[os.path.basename(fname)] = (zi, local_off, csize, method)
        pos += 46 + fname_len + extra_len + comment_len
    return entries


class ZipImageDB:
    """Read images from zip archives by basename, without extracting.

    memory=True: mmap the zip (page cache — reclaimable, no OOM) and read via
    manual central-directory offsets + zlib.decompress. Slice reads are
    lock-free and zlib releases the GIL, so parallel decode is fast even on
    rotational disks (after the first ~3 min page-cache warm-up).
    """

    def __init__(self, zip_paths: list[str], memory: bool = False):
        self.zip_paths = [p for p in zip_paths if os.path.exists(p)]
        if not self.zip_paths:
            raise FileNotFoundError(f"no zip archives found in: {zip_paths}")
        self._memory = memory
        self._mmaps: list[mmap.mmap] = []
        self._files = []
        self.entries: dict[str, tuple[int, str, int, int, int]] = {}
        for zi, p in enumerate(self.zip_paths):
            if memory:
                f = open(p, "rb")
                self._files.append(f)
                try:
                    os.madvise(f.fileno(), 0, os.path.getsize(p), os.MADV_WILLNEED)
                except Exception:
                    pass
                self._mmaps.append(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
                print(f"[zipdb] mmap {p} ({os.path.getsize(p)/1e9:.1f} GB)", flush=True)
            # index via zipfile (ZIP64-aware); reads use mmap slices in memory mode
            n0 = len(self.entries)
            with zipfile.ZipFile(p) as zf:
                for info in zf.infolist():
                    self.entries[os.path.basename(info.filename)] = (
                        zi, info.filename, info.header_offset, info.compress_size, info.compress_type)
            print(f"[zipdb] indexed {p}: +{len(self.entries) - n0} entries", flush=True)

    def read(self, fname: str):
        """Return decompressed image bytes, or None if missing."""
        hit = self.entries.get(os.path.basename(fname))
        if hit is None:
            return None
        zi, entry, local_off, csize, method = hit
        try:
            if self._memory:
                mm = self._mmaps[zi]
                hdr = mm[local_off:local_off + 30]
                nlen, elen = struct.unpack("<HH", hdr[26:30])
                data_off = local_off + 30 + nlen + elen
                raw = mm[data_off:data_off + csize]
                if method == 8:                       # DEFLATE (raw, no zlib header)
                    return zlib.decompressobj(-15).decompress(raw)
                return raw
            with zipfile.ZipFile(self.zip_paths[zi]) as zf:
                return zf.read(entry)
        except Exception:
            return None

    def open_image(self, fname: str):
        raw = self.read(fname)
        if raw is None:
            return None
        try:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None

    def close(self):
        for m in self._mmaps:
            try:
                m.close()
            except Exception:
                pass
        for f in self._files:
            f.close()
