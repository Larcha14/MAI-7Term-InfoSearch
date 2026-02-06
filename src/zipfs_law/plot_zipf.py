import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv


def fit_zipf(rank, freq, r_min=1, r_max=None):

    if r_max is None:
        r_max = rank.max()

    mask = (rank >= r_min) & (rank <= r_max) & (freq > 0)
    x = np.log(rank[mask])
    y = np.log(freq[mask])

    b, a = np.polyfit(x, y, 1)
    s = -b
    C = np.exp(a)
    return C, s


def main() -> int:
    load_dotenv()

    export_dir = os.getenv("EXPORT_DIR", "/exports")
    zipf_file = os.getenv("ZIPF_FILE", "zipf.csv")
    csv_path = os.path.join(export_dir, zipf_file)

    if not os.path.exists(csv_path):
        print(f"[ERROR] zipf csv not found: {csv_path}", file=sys.stderr)
        return 2

    df = pd.read_csv(csv_path)

    for col in ("rank", "freq"):
        if col not in df.columns:
            print(f"[ERROR] missing column: {col}", file=sys.stderr)
            return 3

    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce")
    df = df.dropna(subset=["rank", "freq"]).sort_values("rank")

    if df.empty:
        print("[ERROR] empty after parsing", file=sys.stderr)
        return 4

    rank = df["rank"].to_numpy()
    freq = df["freq"].to_numpy()

    # --- Теоретический Zipf: s=1, C=f(1) ---
    f1 = freq[0]
    zipf_theory = f1 * (rank ** -1.0)

    C_fit, s_fit = fit_zipf(rank, freq, r_min=10, r_max=min(10000, int(rank.max())))
    zipf_fit = C_fit * (rank ** (-s_fit))

    plt.figure()
    plt.loglog(rank, freq, marker=".", linestyle="none", label="Corpus data")
    plt.loglog(rank, zipf_theory, linestyle="-", label="Zipf: s=1, C=f(1)")
    plt.loglog(rank, zipf_fit, linestyle="--", label=f"Fit: s={s_fit:.3f}")

    plt.xlabel("rank (log)")
    plt.ylabel("frequency (log)")
    plt.title("Zipf plot with Zipf law overlay")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(export_dir, "zipf_plot.png")
    plt.savefig(out_path, dpi=200)

    print(f"[INFO] saved: {out_path}")
    print(f"[INFO] fitted s = {s_fit:.6f}, C = {C_fit:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
