"""Extract playable agents from the public notebooks in research/nb/.

These become local sparring partners. Beating `starter` proves nothing -- the
ladder is full of strong agents, so the eval pool has to be too.

Handles the three ways public notebooks ship an agent:
  1. a `%%writefile main.py` / `%%agentfile` cell  -> source is the cell body
  2. a base85+zlib blob in `_AGENT_B85_PARTS`      -> decode
  3. a base64+gzip blob assigned to a string       -> decode

Nothing here is redistributed; the extracted files stay local and are listed in
.gitignore. Provenance is recorded in gauntlet/PROVENANCE.md.
"""
import ast
import base64
import gzip
import json
import os
import re
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.environ.get("NB_DIR") or os.path.join(HERE, "research", "nb")
OUT = os.environ.get("GAUNTLET_OUT") or os.path.join(HERE, "gauntlet")


def _write(name, src, provenance, index):
    if isinstance(src, bytes):
        src = src.decode("utf-8", "replace")
    if "def agent" not in src:
        return False
    path = os.path.join(OUT, name + ".py")
    with open(path, "w") as f:
        f.write(src)
    index.append((name, provenance, len(src)))
    print(f"  wrote {name}.py  ({len(src):,} bytes)")
    return True


def from_magic_cell(src):
    """`%%writefile main.py` / `%%agentfile` -> the rest of the cell."""
    first, _, body = src.partition("\n")
    if first.strip().startswith(("%%writefile", "%%agentfile")):
        return body
    return None


def from_b85_parts(src):
    """`_AGENT_B85_PARTS = [...]` -> zlib.decompress(b85decode(''.join(parts)))."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not any("B85" in n or "B85_PARTS" in n for n in names):
            continue
        try:
            parts = ast.literal_eval(node.value)
        except Exception:
            continue
        if not isinstance(parts, (list, tuple)):
            continue
        try:
            return zlib.decompress(base64.b85decode("".join(parts).encode("ascii")))
        except Exception:
            return None
    return None


def from_b64_gzip(src):
    """Any long base64 string literal that gunzips into Python source."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    best = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) > 5000:
            blob = node.value.strip()
            if not re.fullmatch(r"[A-Za-z0-9+/=\s]+", blob):
                continue
            for dec in (gzip.decompress, zlib.decompress):
                try:
                    raw = dec(base64.b64decode(blob.encode("ascii")))
                except Exception:
                    continue
                if b"def agent" in raw and (best is None or len(raw) > len(best)):
                    best = raw
    return best


def from_source_string(src):
    """`AGENT_SOURCE = '...'` holding the literal source."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not any("SOURCE" in n or "AGENT" in n for n in names):
            continue
        try:
            val = ast.literal_eval(node.value)
        except Exception:
            continue
        if isinstance(val, str) and "def agent" in val and len(val) > 2000:
            return val
    return None


EXTRACTORS = (from_magic_cell, from_b85_parts, from_b64_gzip, from_source_string)


def main():
    os.makedirs(OUT, exist_ok=True)
    index = []
    for d in sorted(os.listdir(NB_DIR)):
        nbdir = os.path.join(NB_DIR, d)
        if not os.path.isdir(nbdir):
            continue
        for fn in os.listdir(nbdir):
            if not fn.endswith(".ipynb"):
                continue
            nb = json.load(open(os.path.join(nbdir, fn)))
            author = d.split("_")[0]
            best = None
            for cell in nb.get("cells", []):
                if cell.get("cell_type") != "code":
                    continue
                src = "".join(cell.get("source", []))
                if len(src) < 1500:
                    continue
                for ex in EXTRACTORS:
                    try:
                        got = ex(src)
                    except Exception:
                        got = None
                    if got and b"def agent" in (got if isinstance(got, bytes)
                                                else got.encode("utf-8", "replace")):
                        if best is None or len(got) > len(best):
                            best = got
                        break
            if best is not None:
                print(f"{d}:")
                _write(author, best, f"{d}/{fn}", index)

    with open(os.path.join(OUT, "PROVENANCE.md"), "w") as f:
        f.write("# Gauntlet provenance\n\n")
        f.write("Local sparring partners extracted from public Kaggle notebooks.\n")
        f.write("**Not redistributed** -- these stay local and are gitignored.\n\n")
        f.write("| agent | source notebook | bytes |\n|---|---|---:|\n")
        for name, prov, n in sorted(index):
            f.write(f"| `{name}` | {prov} | {n:,} |\n")
    print(f"\n{len(index)} agents -> {OUT}/")


if __name__ == "__main__":
    main()
