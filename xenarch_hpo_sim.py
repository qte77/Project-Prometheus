"""
xenarch_hpo_sim.py  -  smart hyperparameter search, kept deliberately small.

Method: RANDOM SEARCH + SUCCESSIVE HALVING (a mini-Hyperband).
  - Grid search spends the same compute on every config, including the hopeless ones.
  - Here we sample configs at random, train them ALL a little, throw away the worst
    two thirds, train the survivors longer, halve again, and so on.
    Compute goes where the results are.

Target model: a tiny conv autoencoder on synthetic lunar tiles - same family as the
real Xenarch VAE (strided-conv encoder -> latent bottleneck -> transposed-conv decoder),
just small enough that one trial takes ~1 s on CPU. To tune the real model instead,
replace `evaluate()` with a call to xenarch_mk15_script's pipeline.

Run:  python xenarch_hpo_sim.py
Out:  results/hpo_sim.csv , results/hpo_sim.png
"""

import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------------------------
# synthetic data: smooth terrain + a few craters + noise, normalised
# ----------------------------------------------------------------------
def make_tile():
    t = np.zeros((16, 16), np.float32)
    gx, gy = rng.uniform(-.25, .25, 2)
    base = rng.uniform(.35, .55)
    fx, ph = rng.uniform(1, 3), rng.uniform(0, 6.28)
    yy, xx = np.mgrid[0:16, 0:16]
    t += base + gx * (xx / 16 - .5) + gy * (yy / 16 - .5) + .08 * np.sin(fx * (xx + yy) / 3 + ph)
    for _ in range(rng.integers(1, 4)):
        cx, cy = rng.uniform(2, 14, 2)
        r, depth = rng.uniform(1.6, 5.0), rng.uniform(.15, .43)
        d = np.hypot(xx - cx, yy - cy) / r
        t -= depth * (1 - d ** 2) * (d < 1)
        t += depth * .5 * np.clip(1.35 - d, 0, None) / .35 * ((d >= 1) & (d < 1.35))
    t += rng.normal(0, .05, t.shape).astype(np.float32)
    t = (t - t.min()) / (t.max() - t.min() + 1e-6)
    return t

def make_set(n):
    return torch.from_numpy(np.stack([make_tile() for _ in range(n)])[:, None]).float()

TRAIN = make_set(160)
VAL = make_set(96)
torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))

# ----------------------------------------------------------------------
# the model under test
# ----------------------------------------------------------------------
class AE(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, 2, 1), nn.ReLU(),
        )                                   # -> 16 x 4 x 4
        self.to_z = nn.Linear(16 * 4 * 4, latent)
        self.from_z = nn.Linear(latent, 16 * 4 * 4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, 2, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.enc(x).flatten(1)
        z = self.to_z(h)
        h = self.from_z(z).view(-1, 16, 4, 4)
        return self.dec(h)

def evaluate(cfg, epochs):
    """Train a fresh model for `epochs`, return validation MSE (lower is better)."""
    torch.manual_seed(SEED)
    net = AE(cfg["latent"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    lossf = nn.MSELoss()
    n, bs = TRAIN.shape[0], cfg["batch"]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            xb = TRAIN[perm[i:i + bs]]
            opt.zero_grad()
            loss = lossf(net(xb), xb)
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        vl = lossf(net(VAL), VAL).item()
    return vl

# ----------------------------------------------------------------------
# search space  -  four knobs, sampled at random
# ----------------------------------------------------------------------
def sample_cfg():
    return dict(
        lr=float(10 ** rng.uniform(-4.0, -1.5)),                       # 1e-4 .. ~3e-2
        latent=int(rng.choice([2, 3, 4, 6, 8, 12, 16, 24])),
        batch=int(rng.choice([8, 16, 32])),
        wd=float(10 ** rng.uniform(-6, -3)) if rng.random() < .5 else 0.0,
    )

# ----------------------------------------------------------------------
# successive halving
# ----------------------------------------------------------------------
N = 18                       # configs sampled  (18 -> 6 -> 2 survivors)
ETA = 3                      # keep 1 / ETA each rung
RUNGS = [5, 15, 45]          # cumulative epochs trained at each rung
CONFIRM = 100                # epochs for the final winner re-check

def main():
    t0 = time.time()
    Path("results").mkdir(exist_ok=True)

    configs = [sample_cfg() for _ in range(N)]
    alive = list(range(N))
    history = []

    print(f"random search + successive halving  -  {N} configs, rungs {RUNGS}\n", flush=True)
    for r, ep in enumerate(RUNGS):
        scored = []
        for idx in alive:
            vl = evaluate(configs[idx], ep)
            scored.append((vl, idx))
            history.append({"rung": r, "epochs": ep, "val_loss": vl, **configs[idx]})
            print(".", end="", flush=True)
        scored.sort()
        keep = max(1, len(alive) // ETA)
        survivors = [i for _, i in scored[:keep]]
        print(f"\nrung {r}: trained {len(scored):2d} configs @ {ep:3d} epochs "
              f"-> keep {keep}  (best val_loss {scored[0][0]:.5f})", flush=True)
        alive = survivors

    best_idx = alive[0]
    best = configs[best_idx]
    best_vl = evaluate(best, CONFIRM)      # confirm the winner with a longer run
    dt = time.time() - t0

    print("\n" + "=" * 60)
    print("BEST CONFIG")
    print("=" * 60)
    print(f"  learning_rate : {best['lr']:.2e}")
    print(f"  latent_dim    : {best['latent']}")
    print(f"  batch_size    : {best['batch']}")
    print(f"  weight_decay  : {best['wd']:.2e}")
    print(f"  val_loss @{CONFIRM:<3} : {best_vl:.5f}")

    spent = sum(len([h for h in history if h['rung'] == r]) * ep for r, ep in enumerate(RUNGS))
    print(f"\n  epoch-units spent : {spent}   (wall time {dt:.1f}s)")
    print(f"  a flat grid at {RUNGS[-1]} epochs/config would have tested only "
          f"~{spent // RUNGS[-1]} configs for the same cost")

    # full leaderboard from the deepest rung each config reached
    deepest = {}
    for h in history:
        k = (h['lr'], h['latent'], h['batch'], h['wd'])
        if k not in deepest or h['rung'] >= deepest[k]['rung']:
            deepest[k] = h
    board = sorted(deepest.values(), key=lambda h: h['val_loss'])
    print("\n  rank  val_loss   lr        latent  batch  wd        rung")
    for i, h in enumerate(board, 1):
        print(f"  {i:>3}   {h['val_loss']:.5f}   {h['lr']:.1e}   {h['latent']:>4}   "
              f"{h['batch']:>4}   {h['wd']:.1e}   {h['rung']}")

    with open("results/hpo_sim.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rung", "epochs", "val_loss", "lr", "latent", "batch", "wd"])
        w.writeheader()
        w.writerows(history)
    print("\n  wrote results/hpo_sim.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for r in range(len(RUNGS)):
            pts = [h for h in history if h["rung"] == r]
            ax.scatter([p["lr"] for p in pts], [p["val_loss"] for p in pts],
                       s=[18 + 6 * p["latent"] for p in pts], alpha=.7,
                       label=f"rung {r} ({RUNGS[r]} ep)")
        ax.scatter([best["lr"]], [best_vl], marker="*", s=320, c="#c0392b",
                   zorder=5, label="best")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("learning rate"); ax.set_ylabel("validation MSE")
        ax.set_title("Successive halving  -  point size = latent dim")
        ax.legend(fontsize=8); ax.grid(alpha=.25, which="both")
        fig.tight_layout(); fig.savefig("results/hpo_sim.png", dpi=140)
        print("  wrote results/hpo_sim.png")
    except Exception as e:
        print(f"  (plot skipped: {e})")

if __name__ == "__main__":
    main()
