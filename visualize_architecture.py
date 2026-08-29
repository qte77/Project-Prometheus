"""
Render the Xenarch Stable Convolutional VAE architecture as a labelled diagram.

Shows structural depth (layer count) and size (channels / spatial resolution /
parameter count) of every stage, from the 1x256x256 input chip through the
56-D latent bottleneck and back out to the reconstruction.

Output: results/xenarch_vae_architecture.png
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

# --- pull exact parameter counts straight from the model definition ----------
try:
    from xenarch_mk15_script import StableConvolutionalVAE

    _m = StableConvolutionalVAE(latent_dim=56, input_size=256)
    TOTAL_PARAMS = sum(p.numel() for p in _m.parameters())
    _named = {n: p.numel() for n, p in _m.named_parameters()}

    def _grp(prefix):
        return sum(v for k, v in _named.items() if k.startswith(prefix))

    P_ENC_CONV = _grp("encoder_conv")
    P_MU = _grp("fc_mu")
    P_LOGVAR = _grp("fc_logvar")
    P_DEC_IN = _grp("decoder_input")
    P_DEC_CONV = _grp("decoder_conv")
except Exception as e:  # torch missing / import failure -> fall back to static numbers
    print(f"[warn] could not import model ({e}); using static parameter counts")
    P_ENC_CONV, P_MU, P_LOGVAR, P_DEC_IN, P_DEC_CONV = 388800, 3670072, 3670072, 3735552, 388033
    TOTAL_PARAMS = P_ENC_CONV + P_MU + P_LOGVAR + P_DEC_IN + P_DEC_CONV


def fmt(n):
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    if n >= 1e3:
        return f"{n / 1e3:.1f} K"
    return str(int(n))


# --- stage tables -----------------------------------------------------------
# (label, output-shape text, spatial size, channels, block params)
encoder = [
    ("Input chip",                       "1 x 256 x 256",   256,   1, 0),
    ("Conv 3x3  s2\nBatchNorm . ReLU",   "32 x 128 x 128",  128,  32, 384),
    ("Conv 3x3  s2\nBatchNorm . ReLU",   "64 x 64 x 64",     64,  64, 18_624),
    ("Conv 3x3  s2\nBatchNorm . ReLU",   "128 x 32 x 32",    32, 128, 74_112),
    ("Conv 3x3  s2\nBatchNorm . ReLU",   "256 x 16 x 16",    16, 256, 295_680),
]
decoder = [
    ("ConvT 3x3  s2\nBatchNorm . ReLU",  "128 x 32 x 32",    32, 128, 295_296),
    ("ConvT 3x3  s2\nBatchNorm . ReLU",  "64 x 64 x 64",     64,  64, 73_920),
    ("ConvT 3x3  s2\nBatchNorm . ReLU",  "32 x 128 x 128",  128,  32, 18_528),
    ("ConvT 3x3  s2\nSigmoid",           "1 x 256 x 256",   256,   1, 289),
]

# colour by channel count (log scale)
cmap = plt.get_cmap("viridis")
cnorm = matplotlib.colors.LogNorm(vmin=1, vmax=256)


def hbox(spatial):
    """block half-height, compressed so 256 vs 16 stays readable"""
    return 0.8 + 2.9 * (np.sqrt(spatial) / np.sqrt(256))


fig, ax = plt.subplots(figsize=(20, 10.5))
ax.set_xlim(0, 32)
ax.set_ylim(-8.2, 6.4)
ax.axis("off")

x = 1.0
step = 2.15
bw = 1.15
centers = []  # (x, top, bottom, channels) for arrows

def draw_stage(label, shape, spatial, ch, params, x):
    h = hbox(spatial)
    face = cmap(cnorm(max(ch, 1)))
    txtcol = "white" if cnorm(max(ch, 1)) > 0.55 else "#111111"
    ax.add_patch(Rectangle((x, -h), bw, 2 * h, facecolor=face,
                           edgecolor="#222222", linewidth=1.3, zorder=3))
    # channel count inside the block
    ax.text(x + bw / 2, 0, f"{ch}\nch", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=txtcol, zorder=4)
    # op label above
    ax.text(x + bw / 2, h + 0.35, label, ha="center", va="bottom",
            fontsize=8.2, linespacing=1.25)
    # shape + params below
    tail = f"{shape}" if params == 0 else f"{shape}\n{fmt(params)} params"
    ax.text(x + bw / 2, -h - 0.35, tail, ha="center", va="top",
            fontsize=8.0, color="#333333", linespacing=1.3)
    centers.append((x, x + bw))
    return x + bw


# encoder
for (label, shape, sp, ch, pr) in encoder:
    xr = draw_stage(label, shape, sp, ch, pr, x)
    x += step
enc_right = xr

# ---- bottleneck --------------------------------------------------------------
bx = x + 0.15
ax.text(bx + 0.35, 3.55, "Flatten\n65 536", ha="center", fontsize=8.0,
        style="italic", color="#333333")

# mu / logvar dense heads
for dy, name, pr in [(1.5, r"$\mu$  (56)", P_MU), (-1.5, r"$\log\sigma^{2}$  (56)", P_LOGVAR)]:
    ax.add_patch(Rectangle((bx, dy - 0.55), 0.85, 1.1, facecolor="#f2c14e",
                           edgecolor="#222222", linewidth=1.2, zorder=3))
    ax.text(bx + 0.42, dy, name, ha="center", va="center", fontsize=8.2, zorder=4)
    ax.text(bx + 0.42, dy + (0.95 if dy > 0 else -0.95), f"Linear\n{fmt(pr)}",
            ha="center", va="center", fontsize=7.3, color="#333333")

# sampled latent z
zx = bx + 1.9
ax.add_patch(Circle((zx + 0.35, 0), 0.62, facecolor="#e0554e",
                    edgecolor="#222222", linewidth=1.3, zorder=3))
ax.text(zx + 0.35, 0, "z\n56", ha="center", va="center", fontsize=8.5,
        fontweight="bold", color="white", zorder=4)
ax.text(zx + 0.35, -1.15, "reparameterise\n$z=\\mu+\\varepsilon\\,\\sigma$",
        ha="center", va="top", fontsize=7.4, color="#333333")

# decoder_input dense -> reshape
dix = zx + 1.7
ax.add_patch(Rectangle((dix, -0.7), 0.9, 1.4, facecolor="#f2c14e",
                       edgecolor="#222222", linewidth=1.2, zorder=3))
ax.text(dix + 0.45, 0, "Linear\n56 ->\n65 536", ha="center", va="center", fontsize=7.2, zorder=4)
ax.text(dix + 0.45, 1.15, f"{fmt(P_DEC_IN)}", ha="center", fontsize=7.3, color="#333333")
ax.text(dix + 0.45, -1.15, "reshape\n256 x 16 x 16", ha="center", va="top",
        fontsize=7.3, color="#333333")

# arrows across the bottleneck
def arrow(x0, x1, y0=0, y1=0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=13, linewidth=1.1, color="#555555", zorder=2))

arrow(enc_right, bx)
arrow(bx + 0.85, zx + 0.35 - 0.6, 1.5, 0.35)
arrow(bx + 0.85, zx + 0.35 - 0.6, -1.5, -0.35)
arrow(zx + 0.35 + 0.62, dix)
arrow(dix + 0.9, dix + 1.35)

# decoder
x = dix + 1.9
dec_left = x
first = True
for (label, shape, sp, ch, pr) in decoder:
    if not first:
        pass
    xr = draw_stage(label, shape, sp, ch, pr, x)
    x += step
    first = False

# straight-through arrows between conv blocks (encoder + decoder)
for (l, r) in zip(centers[:-1], centers[1:]):
    # skip the encoder->bottleneck and bottleneck->decoder gaps (handled above)
    gap = r[0] - l[1]
    if gap < step:  # adjacent blocks only
        arrow(l[1], r[0])

# --- headings & annotations -------------------------------------------------
ax.text(16, 6.0, "Xenarch  -  Stable Convolutional VAE", ha="center",
        fontsize=17, fontweight="bold")
ax.text(16, 5.15,
        "11 learnable layers   .   4 strided conv (down)  +  3 dense  +  4 transposed conv (up)   "
        ".   latent dim 56   .   input 1 x 256 x 256   .   "
        f"{fmt(TOTAL_PARAMS)} trainable parameters",
        ha="center", fontsize=10, color="#333333")

ax.text(7.0, -6.45, "ENCODER  (recognition network)", ha="center", fontsize=10,
        fontweight="bold", color="#2a2a2a")
ax.text(25.0, -6.45, "DECODER  (generative network)", ha="center", fontsize=10,
        fontweight="bold", color="#2a2a2a")
ax.annotate("", xy=(12.4, -6.45), xytext=(1.6, -6.45),
            arrowprops=dict(arrowstyle="-", color="#999999"))
ax.annotate("", xy=(30.4, -6.45), xytext=(19.6, -6.45),
            arrowprops=dict(arrowstyle="-", color="#999999"))

# parameter-budget bar
budget = [("Encoder conv", P_ENC_CONV, "#3b7dd8"),
          (r"$\mu$ + $\log\sigma^{2}$ heads", P_MU + P_LOGVAR, "#e0554e"),
          ("decoder_input", P_DEC_IN, "#f2c14e"),
          ("Decoder conv", P_DEC_CONV, "#4c9f70")]
bx0, bx1, by = 2.0, 30.0, -7.5
acc = 0
for name, val, col in budget:
    w = (val / TOTAL_PARAMS) * (bx1 - bx0)
    ax.add_patch(Rectangle((bx0 + acc, by - 0.22), w, 0.44, facecolor=col,
                           edgecolor="white", linewidth=0.8, zorder=3))
    lbl = f"{name}  -  {fmt(val)}  ({100*val/TOTAL_PARAMS:.0f}%)"
    if w > 3.0:
        ax.text(bx0 + acc + w / 2, by, lbl, ha="center", va="center",
                fontsize=6.8, color="white", fontweight="bold", zorder=4)
    else:
        ax.text(bx0 + acc + w / 2, by + 0.62, lbl, ha="center", va="center",
                fontsize=6.8, color="#333333", zorder=4)
    acc += w
ax.text(bx0, by - 0.75, "parameter budget", fontsize=7.6, color="#555555")

# channel colour legend
sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
cax = fig.add_axes([0.905, 0.30, 0.012, 0.4])
cb = fig.colorbar(sm, cax=cax)
cb.set_label("feature-map channels", fontsize=8)
cb.ax.tick_params(labelsize=7)

out = Path("results") / "xenarch_vae_architecture.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved {out}  ({TOTAL_PARAMS:,} params)")
