import numpy as np
from scipy.special import jv
from scipy.optimize import brentq
import json
import os
from pathlib import Path

CACHE_FILE = Path(__file__).resolve().parent / "bessel_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {CACHE_FILE} is empty or corrupted. Reinitializing...")
                return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def compute_bessel_zeros(v, start_n, end_n):
    zeros = []
    step = np.pi / 2
    x = 0.1

    prev_zeros = get_bessel_zeros_from_cache(v, start_n)
    if prev_zeros:
        zeros.extend(prev_zeros)
        x = prev_zeros[-1] + step

    while len(zeros) < end_n:
        if jv(v, x) * jv(v, x + step) < 0:
            root = brentq(lambda x_val: jv(v, x_val), x, x + step)
            zeros.append(float(root))
        x += step

    return zeros[start_n:] 

def get_bessel_zeros(dimension, n):
    n = int(n)
    v = (dimension - 2) / 2
    v_str = f"{v:.6f}"
    cache = load_cache()

    if v_str in cache:
        existing = cache[v_str]
        existing_zeros = existing["zeros"]
        max_n = int(existing["max_n"])
        if n <= max_n:
            return np.array(existing_zeros[:n])
        else:
            new_zeros = compute_bessel_zeros(v, max_n, n)
            combined_zeros = existing_zeros + new_zeros
            cache[v_str] = {
                "max_n": n,
                "zeros": list(map(float, combined_zeros))
            }
            save_cache(cache)
            return np.array(combined_zeros[:n])
    else:
        zeros = compute_bessel_zeros(v, 0, n)
        cache[v_str] = {
            "max_n": n,
            "zeros": list(map(float, zeros))
        }
        save_cache(cache)
        return np.array(zeros[:n])

def get_bessel_zeros_from_cache(v, n):
    cache = load_cache()
    v_str = f"{v:.6f}"
    if v_str in cache and cache[v_str]["max_n"] >= n:
        return cache[v_str]["zeros"][:n]
    return []

get_bessel_zeros(1000, 10)
