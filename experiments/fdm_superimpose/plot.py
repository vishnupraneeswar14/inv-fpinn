from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from fpinns.fdm import FracSDOF

ROOT = Path(__file__).resolve().parents[2]
PHYS_CONFIG = ROOT / "src" / "fpinns" / "config.yaml"
EXP_CONFIG = Path(__file__).resolve().parent / "experiment_config.yaml"
OUT_DIR = Path(__file__).resolve().parent

phys = yaml.safe_load(PHYS_CONFIG.read_text())["physics"]
exp = yaml.safe_load(EXP_CONFIG.read_text())

mode = exp["mode"]
assert mode in ("alpha", "tau"), f"mode must be alpha or tau, got {mode}"
max_points = exp["max_points"]
T = exp["T"]

m, k, c = phys["m"], phys["k"], phys["c"]
x0, v0 = phys["x0"], phys["v0"]
dt, tau = phys["dt"], phys["tau_actual"]
alpha = phys["alpha_actual"]
freq = yaml.safe_load(PHYS_CONFIG.read_text())["model"]["freq"]

if mode == "alpha":
    sweep = np.linspace(exp["alpha"]["min"], exp["alpha"]["max"], exp["alpha"]["points"])
    fixed = tau
else:
    sweep = np.linspace(exp["tau"]["min"], exp["tau"]["max"], exp["tau"]["points"])
    fixed = alpha

t = np.arange(0, T, dt)
F = phys["force_mag"] * np.sin(freq * t)

chunks = [sweep[i:i + max_points] for i in range(0, len(sweep), max_points)]
for chunk in chunks:
    start, end = f"{chunk[0]:.2f}", f"{chunk[-1]:.2f}"
    fig, ax = plt.subplots(figsize=(10, 6))
    for value in chunk:
        if mode == "alpha":
            u, _ = FracSDOF(m, k, c, dt, F, x0, v0, T, float(value), fixed)
            label = f"alpha={value:.2f}"
        else:
            u, _ = FracSDOF(m, k, c, dt, F, x0, v0, T, fixed, float(value))
            label = f"tau={value:.2f}"
        ax.plot(t, u, label=label)
    ax.legend(ncol=2, fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (mm)")
    ax.set_title(f"FDM response, {mode} {start}-{end}, T={T}")
    fig.tight_layout()
    out = OUT_DIR / f"fdm_T_{T}_{mode}_{start}-{end}.jpg"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")