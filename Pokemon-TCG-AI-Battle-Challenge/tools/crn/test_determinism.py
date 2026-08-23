"""Verify the CRN harness produces byte-identical games for a fixed seed.

Runs the same deterministic policy through CrnBattleStart twice with the same
seed, then once with a different seed, and hashes the full JSON observation
stream each time. Same seed must match exactly; different seed must not.
"""
import ctypes
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "agent"))

from meta_decks import GARCHOMP, GRIMMSNARL  # noqa: E402


class StartData(ctypes.Structure):
    _fields_ = [("battlePtr", ctypes.c_void_p),
                ("errorPlayer", ctypes.c_int),
                ("errorType", ctypes.c_int)]


class SerialData(ctypes.Structure):
    _fields_ = [("json", ctypes.c_char_p),
                ("data", ctypes.POINTER(ctypes.c_ubyte)),
                ("count", ctypes.c_int),
                ("selectPlayer", ctypes.c_int)]


lib = ctypes.cdll.LoadLibrary(os.path.join(HERE, "libcrn.so"))
lib.GameInitialize()
lib.CrnBattleStart.restype = StartData
lib.CrnBattleStart.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint]
lib.BattleFinish.argtypes = [ctypes.c_void_p]
lib.GetBattleData.restype = SerialData
lib.GetBattleData.argtypes = [ctypes.c_void_p]
lib.Select.restype = ctypes.c_int
lib.Select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]


def play(seed, deck0, deck1, max_steps=4000):
    """Deterministic policy (always the first legal option); hash the observation stream."""
    cards = (ctypes.c_int * 120)(*(list(deck0) + list(deck1)))
    start = lib.CrnBattleStart(cards, ctypes.c_uint(seed))
    if start.errorPlayer >= 0:
        raise SystemExit(f"deck error player={start.errorPlayer} type={start.errorType}")
    ptr = start.battlePtr
    h = hashlib.sha256()
    steps = 0
    result = None
    try:
        while steps < max_steps:
            sd = lib.GetBattleData(ptr)
            if not sd.json:
                break
            raw = ctypes.string_at(sd.json)
            h.update(raw)
            obs = json.loads(raw.decode("utf-8", "replace"))
            cur = obs.get("current") or {}
            if cur.get("result", -1) >= 0:
                result = cur["result"]
                break
            sel = obs.get("select")
            if sel is None:
                break
            n = len(sel.get("option") or [])
            if n == 0:
                break
            count = sel.get("maxCount", 1) or 1
            choice = list(range(min(count, n)))
            arr = (ctypes.c_int * len(choice))(*choice)
            if lib.Select(ptr, arr, len(choice)) != 0:
                break
            steps += 1
    finally:
        lib.BattleFinish(ptr)
    return h.hexdigest(), steps, result


if __name__ == "__main__":
    a = play(12345, GARCHOMP, GRIMMSNARL)
    b = play(12345, GARCHOMP, GRIMMSNARL)
    c = play(999, GARCHOMP, GRIMMSNARL)
    print(f"seed 12345 run A: {a[0][:16]}  steps={a[1]} result={a[2]}")
    print(f"seed 12345 run B: {b[0][:16]}  steps={b[1]} result={b[2]}")
    print(f"seed   999 run C: {c[0][:16]}  steps={c[1]} result={c[2]}")
    same = a[0] == b[0]
    diff = a[0] != c[0]
    print(f"\nreproducible (A==B): {same}")
    print(f"seed actually matters (A!=C): {diff}")
    print("RESULT:", "PASS - common random numbers available" if (same and diff) else "FAIL")
