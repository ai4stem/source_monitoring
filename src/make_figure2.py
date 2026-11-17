"""
make_figure2.py  —  Figure 2 (2x2 results): correctness legibility with stability
controlled, its dependence on the stability regime, and its (non-)dependence on scale.
=====================================================================================
Draws directly from the analysis JSONs, so every plotted number is traceable.
Stability is the MEANING-LEVEL leave-one-paraphrase-out support
(recompute_support_semantic.py; variant "anchor", correctness-blind).

  a  correctness AUROC on pairs matched on support and on source dataset, per model, 95% Hanley-McNeil CI, with the
     support-only AUROC on the same matched set (0.50) as the reference
     and the uncontrolled AUROC on all facts for comparison
  b  correctness AUROC among facts whose six answers all agree (unanimous; 95% CI
     shown, since the incorrect class is small) versus support-matched pairs drawn
     from the remaining facts (a like-for-like comparison: support is fixed in both)
  c-e  uncontrolled, matched and unanimous AUROC against model size (log2), OLS fit +
     95% confidence band, slope per doubling with 95% CI and TOST equivalence P
        <- outputs/analysis_2x2_primary.json, outputs/scale_correctness_semantic.json
           (the fit is recomputed here and checked against the scale JSON)

No model re-runs. Output: vector PDF + SVG (editable text) + 600 dpi PNG.

Usage (from src/):
    python make_figure2.py
    python make_figure2.py --variant gold --name figure2_gold_sensitivity
"""
from __future__ import annotations
import argparse, json, math, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------- style
MM = 1 / 25.4
INK, INK2, INK3 = "#1a1a1a", "#555555", "#8c8c8c"
GRID = "#e6e6e6"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans",
                        "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.5, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "mathtext.default": "regular",
})

# Okabe-Ito subset (validated colour-blind safe); shape is a redundant family cue.
FAMILY_STYLE = {
    "Qwen3":   ("#0072b2", "o"),
    "Llama":   ("#d55e00", "s"),
    "Phi":     ("#009e73", "^"),
    "Mistral": ("#cc79a7", "D"),
    "Gemma":   ("#e69f00", "v"),
}
# key -> (display name, family, size in billions of parameters); order = by size
MODELS = [
    ("llama32-1b",       "Llama 3.2 1B",     "Llama",   1.0),
    ("qwen3-1.7b",       "Qwen3 1.7B",       "Qwen3",   1.7),
    ("llama32-3b",       "Llama 3.2 3B",     "Llama",   3.0),
    ("phi4-mini",        "Phi-4-mini",       "Phi",     3.8),
    ("phi3-mini-legacy", "Phi-3-mini",       "Phi",     3.8),
    ("qwen3-4b",         "Qwen3 4B",         "Qwen3",   4.0),
    ("mistral-7b",       "Mistral 7B",       "Mistral", 7.0),
    ("qwen3-8b",         "Qwen3 8B",         "Qwen3",   8.0),
    ("llama3-8b-legacy", "Llama 3 8B",       "Llama",   8.0),
    ("gemma3-12b",       "Gemma 3 12B",      "Gemma",  12.0),
    ("mistral-nemo-12b", "Mistral Nemo 12B", "Mistral", 12.0),
    ("qwen3-14b",        "Qwen3 14B",        "Qwen3",  14.0),
    ("qwen3-32b",        "Qwen3 32B",        "Qwen3",  32.0),
]
MS = 4.6  # marker size (pt)


def hanley_mcneil_ci(a, n_pos, n_neg, z=1.96):
    q1 = a / (2 - a); q2 = 2 * a * a / (1 + a)
    se = math.sqrt((a * (1 - a) + (n_pos - 1) * (q1 - a * a)
                    + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg))
    return a - z * se, a + z * se


def ols(sizes, vals, bound=0.02):
    x = np.log2(np.asarray(sizes, float)); y = np.asarray(vals, float)
    r = stats.linregress(x, y); df = len(x) - 2
    tc = stats.t.ppf(0.975, df)
    p_tost = max(1 - stats.t.cdf((r.slope + bound) / r.stderr, df),
                 stats.t.cdf((r.slope - bound) / r.stderr, df))
    resid = y - (r.intercept + r.slope * x)
    s = math.sqrt(float(resid @ resid) / df)

    def band(xg):
        yg = r.intercept + r.slope * xg
        h = tc * s * np.sqrt(1 / len(x) + (xg - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
        return yg, yg - h, yg + h
    return {"slope": r.slope, "ci": (r.slope - tc * r.stderr, r.slope + tc * r.stderr),
            "p": r.pvalue, "tost_p": p_tost, "band": band}


def panel_label(fig, ax, s, x_fig, dy_pt=4):
    """Bold lower-case panel letter at a fixed figure x, just above the axes top."""
    top = ax.get_position().y1 + dy_pt / 72 / fig.get_figheight()
    fig.text(x_fig, top, s, fontsize=9, fontweight="bold", va="bottom", ha="left")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--primary", default=os.path.join(here, "outputs", "analysis_2x2_primary.json"))
    ap.add_argument("--scale", default=os.path.join(here, "outputs", "scale_correctness_semantic.json"))
    ap.add_argument("--variant", default="anchor")
    ap.add_argument("--out_dir", default=os.path.join(here, "..", "figures"))
    ap.add_argument("--name", default="figure2_2x2_results")
    args = ap.parse_args()
    prim = {r["model"]: r for r in json.load(open(args.primary))}
    sem = {k: r[args.variant] for k, r in prim.items()}
    scale = {r["measure"]: r for r in json.load(open(args.scale))
             if r.get("support_variant", args.variant) == args.variant}

    keys = [m[0] for m in MODELS]
    names = [m[1] for m in MODELS]
    fams = [m[2] for m in MODELS]
    sizes = [m[3] for m in MODELS]
    A = [sem[k]["matched"]["truth_AUROC"] for k in keys]             # matched on support and dataset
    N = [sem[k]["matched"]["n_pairs"] for k in keys]
    SUP = [sem[k]["matched"]["support_only_AUROC"] for k in keys]
    OV = [prim[k]["overall"]["truth_AUROC"] for k in keys]           # no control, all facts
    LO = [sem[k]["matched_rest"]["truth_AUROC"] for k in keys]       # answers disagree, matched
    HI = [sem[k]["unanimous"]["truth_AUROC"] for k in keys]          # all six agree, matched on dataset
    NHI = [sem[k]["unanimous"]["n_pairs"] for k in keys]
    n = len(keys)
    ypos = np.arange(n)[::-1] + 1.6          # models, smallest on top
    ymean = 0.0                              # summary row

    fig = plt.figure(figsize=(180 * MM, 150 * MM))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.62, 1.0], hspace=0.50, wspace=0.9,
                          left=0.145, right=0.985, top=0.935, bottom=0.115)
    axa = fig.add_subplot(gs[0, 0:3])
    axb = fig.add_subplot(gs[0, 3:6])
    axs = [fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]),
           fig.add_subplot(gs[1, 4:6])]

    # ------------------------------------------------------------------ a: matched
    ax = axa
    ax.axvline(0.5, color=INK3, lw=0.6, ls=(0, (3, 2)), zorder=1)
    mA = float(np.mean(A))
    ax.axvline(mA, color=INK3, lw=0.5, zorder=1)
    for y, a, npair, sup, fam in zip(ypos, A, N, SUP, fams):
        col, mk = FAMILY_STYLE[fam]
        lo, hi = hanley_mcneil_ci(a, npair, npair)
        ax.plot([lo, hi], [y, y], color=col, lw=1.0, solid_capstyle="butt", zorder=2)
        ax.plot(a, y, mk, ms=MS, mfc=col, mec="white", mew=0.6, zorder=3)
        ax.plot(sup, y, "x", ms=3.2, mec=INK3, mew=0.7, zorder=2)
    for y, ov in zip(ypos, OV):
        ax.plot(ov, y, "o", ms=MS - 0.6, mfc="white", mec=INK3, mew=0.8, zorder=2)
    ax.plot(float(np.mean(OV)), ymean, "o", ms=MS, mfc="white", mec=INK3, mew=0.8, zorder=2)
    sd = float(np.std(A, ddof=1))
    ax.plot([min(A), max(A)], [ymean, ymean], color=INK, lw=1.0, zorder=2)
    ax.plot(mA, ymean, "o", ms=MS + 0.6, mfc=INK, mec="white", mew=0.6, zorder=3)
    ax.plot(float(np.mean(SUP)), ymean, "x", ms=3.2, mec=INK3, mew=0.7)
    ax.axhline(0.8, color=GRID, lw=0.5)
    ax.set_yticks(list(ypos) + [ymean])
    ax.set_yticklabels(names + ["Mean (range)"])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-1.3, n + 2.1)
    ax.set_xlim(0.47, 0.83)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8])
    ax.set_xlabel("Correctness AUROC")
    ax.text(0.505, n + 1.6, "Stability alone\n(support-only), 0.50", fontsize=6,
            color=INK2, va="center", ha="left", linespacing=1.1)
    ax.text(mA + 0.004, n + 1.6, f"Stability-matched\npairs, mean {mA:.3f}", fontsize=6, color=INK2,
            va="center", ha="left", linespacing=1.1)
    ax.text(float(np.mean(OV)) + 0.004, n + 1.6, f"No control,\nmean {np.mean(OV):.3f}", fontsize=6,
            color=INK2, va="center", ha="left", linespacing=1.1)
    panel_label(fig, ax, "a", 0.012)

    # ------------------------------------------------------------------ b: regimes
    ax = axb
    ax.plot([0.5, 0.5], [-1.3, n + 1.0], color=INK3, lw=0.6, ls=(0, (3, 2)), zorder=0)
    for y, lo, hi, k, nh in zip(ypos, LO, HI, keys, NHI):
        rev = hi > lo
        ax.plot([lo, hi], [y, y], color="#c4c4c4", lw=2.2, zorder=1, solid_capstyle="butt")
        cl, ch = hanley_mcneil_ci(hi, nh, nh)                 # CI of the unanimous AUROC
        ax.plot([cl, ch], [y, y], color=INK, lw=0.5, zorder=2)
        for c in (cl, ch):
            ax.plot([c, c], [y - 0.16, y + 0.16], color=INK, lw=0.5, zorder=2)
        ax.plot(hi, y, "o", ms=MS, mfc=INK, mec="white", mew=0.6, zorder=3)
        ax.plot(lo, y, "o", ms=MS - 0.4, mfc="white", mec=INK, mew=0.9, zorder=3)
        if rev:
            pass
    mlo, mhi = float(np.mean(LO)), float(np.mean(HI))
    ax.plot([mlo, mhi], [ymean, ymean], color=INK, lw=1.4, zorder=1)
    ax.plot(mhi, ymean, "o", ms=MS + 0.6, mfc=INK, mec="white", mew=0.6, zorder=3)
    ax.plot(mlo, ymean, "o", ms=MS + 0.2, mfc="white", mec=INK, mew=0.9, zorder=3)
    left, right = (mhi, mlo) if mhi <= mlo else (mlo, mhi)
    ax.text(left - 0.014, ymean, f"{left:.3f}", fontsize=6, va="center", ha="right")
    ax.text(right + 0.014, ymean, f"{right:.3f}", fontsize=6, va="center", ha="left")
    ax.axhline(0.8, color=GRID, lw=0.5)
    ax.set_yticks(list(ypos) + [ymean]); ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    for y in ypos:                                   # faint row guides, both panels
        axa.axhline(y, color=GRID, lw=0.4, zorder=0)
        ax.axhline(y, color=GRID, lw=0.4, zorder=0)
    ax.set_ylim(-1.3, n + 2.1)
    ax.set_xlim(0.44, 0.83)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8])
    ax.set_xlabel("Correctness AUROC within stability regime")
    nrev = sum(h > l for l, h in zip(LO, HI))
    h = [Line2D([], [], marker="o", ls="", ms=MS, mfc=INK, mec="white", mew=0.6,
                label="All six answers agree (unanimous), with 95% CI"),
         Line2D([], [], marker="o", ls="", ms=MS - 0.4, mfc="white", mec=INK, mew=0.9,
                label="Answers disagree, matched pairs")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(-0.02, 1.035), frameon=False,
              handletextpad=0.2, borderaxespad=0, labelspacing=0.3, ncol=1)
    ax.text(0.826, ymean - 0.75, f"Lower when unanimous in {n - nrev} of {n} models", fontsize=6,
            color=INK2, ha="right", va="center")
    panel_label(fig, ax, "b", ax.get_position().x0 - 0.03)

    # ------------------------------------------------------------------ c-e: scale
    spec = [("uncontrolled correctness", OV, "No control (all facts)", "c"),
            ("caliper-matched correctness", A, "Stability-matched", "d"),
            ("unanimous correctness", HI, "All six answers agree", "e")]
    xg = np.linspace(-0.25, 5.25, 200)
    for ax, (mname, vals, title, lab) in zip(axs, spec):
        fit = ols(sizes, vals)
        ref = scale[mname]                           # traceability check vs server JSON
        assert abs(fit["slope"] - ref["slope"]) < 5e-4, (mname, fit["slope"], ref["slope"])
        assert abs(fit["tost_p"] - ref["tost_p"]) < 5e-3, (mname, fit["tost_p"], ref["tost_p"])
        yg, lo, hi = fit["band"](xg)
        ax.fill_between(xg, lo, hi, color="#000000", alpha=0.07, lw=0, zorder=1)
        ax.plot(xg, yg, color=INK, lw=0.9, zorder=2)
        for s, v, fam in zip(sizes, vals, fams):
            col, mk = FAMILY_STYLE[fam]
            ax.plot(math.log2(s), v, mk, ms=MS, mfc=col, mec="white", mew=0.6,
                    zorder=4 if fam == "Phi" else 3)
        ax.set_xlim(-0.35, 5.35); ax.set_ylim(0.49, 0.815)
        ax.set_xticks([0, 1, 2, 3, 4, 5]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
        ax.set_yticks([0.5, 0.6, 0.7, 0.8])
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_xlabel("Parameters (billions, log scale)")
        ax.set_title(title, loc="left", fontweight="bold", pad=21)
        ptxt = f"{fit['tost_p']:.3f}" if fit["tost_p"] >= 0.001 else "< 0.001"
        ax.text(0.0, 1.03,
                f"Slope {fit['slope']:+.3f} per doubling (P = {fit['p']:.3f})\n"
                f"95% CI {fit['ci'][0]:+.3f} to {fit['ci'][1]:+.3f}; "
                f"TOST P {'' if ptxt.startswith('<') else '= '}{ptxt}".replace("-", "−"),
                transform=ax.transAxes, fontsize=6, color=INK2, va="bottom", ha="left",
                linespacing=1.25)
        if ax is axs[0]:
            ax.set_ylabel("Correctness AUROC", labelpad=4)
        else:
            ax.set_yticklabels([])
        panel_label(fig, ax, lab, 0.012 if ax is axs[0] else ax.get_position().x0 - 0.03,
                    dy_pt=21)

    fh = [Line2D([], [], marker=mk, ls="", ms=MS, mfc=col, mec="white", mew=0.6, label=f)
          for f, (col, mk) in FAMILY_STYLE.items()]
    fig.legend(handles=fh, loc="lower center", bbox_to_anchor=(0.565, 0.008), frameon=False,
               ncol=5, handletextpad=0.1, columnspacing=1.6)

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, args.name)
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".svg")
    fig.savefig(base + ".png", dpi=600)
    print(f"saved {base}.pdf / .svg / .png")
    print(f"  a: mean matched AUROC {mA:.3f} (SD {sd:.3f}, range {min(A):.3f}-{max(A):.3f}); "
          f"support-only mean {np.mean(SUP):.3f}")
    print(f"  b: matched non-unanimous {mlo:.3f}, unanimous {mhi:.3f}; lower when unanimous in {n - nrev}/{n}")


if __name__ == "__main__":
    main()
