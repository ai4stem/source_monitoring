"""
make_figure1.py  —  Figure 1 (design schematic): stability probing, the
correctness x stability design, and the stability-controlled read-outs.
=====================================================================================
Writes a hand-laid-out vector schematic as SVG (editable text, 180 mm wide).
PDF/PNG are rendered from the SVG with headless Chrome if it is available:

    python make_figure1.py                 # -> ../figures/figure1_design.svg (+ .pdf/.png)

Everything stated in the schematic mirrors the code:
  6 prompt variants x 1 sampled answer, T = 0.7   probe_knowledge_REVISED.py
  meaning-level LOPO support                      recompute_support_semantic.py
  majority correctness, greedy h (<= 24 tokens)   extract_hidden_2x2.py
  1:1 caliper matching on support (0.05)          analyze_2x2_matched.py
  support terciles                                analyze_2x2.py
  regression on log2(parameters)                  scale_correctness.py
"""
from __future__ import annotations
import argparse, os, shutil, subprocess

W, H = 720, 372            # 4 px per mm at 180 mm width  ->  10 px = 7.1 pt
INK, INK2, INK3 = "#1a1a1a", "#555555", "#8c8c8c"
RULE, PANEL = "#c9c9c6", "#f6f6f4"
COR, COR_L = "#2f6f9f", "#d6e4ef"        # correct: blue (solid / tint)
INC, INC_L = "#c1553b", "#f3dcd5"        # incorrect: vermilion (solid / tint)
ANS_A, ANS_B = "#4a5568", "#ffffff"      # answer identity in the probing grid

o = []  # svg fragments


def t(x, y, s, size=9.5, w="400", fill=INK, anchor="start", style="", extra=""):
    st = f' font-style="{style}"' if style else ""
    o.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" fill="{fill}" '
             f'text-anchor="{anchor}"{st}{extra}>{s}</text>')


def lines(x, y, rows, size=9.5, lh=12, **kw):
    for i, r in enumerate(rows):
        t(x, y + i * lh, r, size=size, **kw)


def rect(x, y, w, h, fill="none", stroke=RULE, sw=0.75, rx=3, extra=""):
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
             f'stroke="{stroke}" stroke-width="{sw}"{extra}/>')


def line(x1, y1, x2, y2, stroke=INK3, sw=0.75, extra=""):
    o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
             f'stroke-width="{sw}"{extra}/>')


def circ(x, y, r, fill, stroke="none", sw=0.75):
    o.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" '
             f'stroke-width="{sw}"/>')


def panel_head(x, letter, title):
    t(x, 13, letter, size=13, w="700")
    t(x + 16, 13, title, size=10.5, w="700")


def flow_arrow(x, y):
    o.append(f'<path d="M{x},{y} h10 m-4,-4 l4,4 l-4,4" fill="none" stroke="{INK3}" '
             f'stroke-width="1" stroke-linecap="round" stroke-linejoin="round"/>')


# =============================================================== panel a
AX = 1
panel_head(AX, "a", "Stability probing")
rect(AX, 26, 236, 40, fill=PANEL)
t(AX + 10, 42, "Factual question, closed book", w="700")
t(AX + 10, 56, "TriviaQA and PopQA; 6,000 facts per model", size=8.5, fill=INK2)

t(AX, 86, "6 prompt variants, one sampled answer each", w="700")
t(AX, 98, "temperature 0.7; bar length = answer length, fill = answer", size=8.5, fill=INK2)

GX, GY, DY = AX + 26, 116, 15
# one answer per prompt variant: bar length = answer length, fill = which answer is given
answers = [("L", "A"), ("S", "A"), ("S", "A"), ("L", "A"), ("S", "B"), ("S", "A")]
HELD = 2
LW, SW = 62, 24
for r, (length, ans) in enumerate(answers):
    y = GY + r * DY
    held = r == HELD
    t(AX + 4, y + 3.2, f"P{r + 1}", size=8.5, fill=INK3 if held else INK2)
    w = LW if length == "L" else SW
    if held:
        rect(GX, y - 4, w, 8, fill="#ffffff", stroke=INK3, sw=0.7, rx=4, extra=' stroke-dasharray="2 2"')
    else:
        rect(GX, y - 4, w, 8, fill=ANS_A if ans == "A" else ANS_B, stroke=ANS_A, sw=0.9, rx=4)
t(GX + SW + 6, GY + HELD * DY + 3, "held out", size=7.6, fill=INK3, style="italic")
# bracket for the other five rows
bx = GX + LW + 8
o.append(f'<path d="M{bx},{GY - 5} h5 v{5 * DY + 10} h-5" fill="none" stroke="{INK3}" stroke-width="0.75"/>')
line(bx + 5, GY + 2.5 * DY, bx + 12, GY + 2.5 * DY)
TX = bx + 16
lines(TX, GY - 2, ["Leave one prompt out: share", "of the other five answers that", "give their most common answer",
                   "(compared by meaning, not by", "wording or length), averaged", "over the six folds"],
      size=8.3, lh=10.6, fill=INK2)
t(TX, GY + 72, "= support, s ∈ [0, 1]", size=9.5, w="700")

# three per-fact quantities
QY = 224
t(AX, QY, "Three quantities per fact", w="700")
rows = [("s", "Stability (support)", "leave-one-prompt-out agreement of answers;", "computed without reference to correctness"),
        ("c", "Correctness", "at least half of the six answers", "contain the reference answer"),
        ("h", "Hidden state", "last token of a separate greedy answer (≤ 24", "tokens); four depths, 40–100% of layers")]
for i, (sym, name, d1, d2) in enumerate(rows):
    y = QY + 12 + i * 42
    rect(AX, y, 236, 36, fill="#ffffff")
    circ(AX + 17, y + 18, 9.5, PANEL, stroke=RULE)
    t(AX + 17, y + 21.5, sym, size=10, w="700", anchor="middle", style="italic")
    t(AX + 34, y + 13, name, size=9, w="700")
    t(AX + 34, y + 23, d1, size=8, fill=INK2)
    t(AX + 34, y + 32, d2, size=8, fill=INK2)

flow_arrow(243, 150)

# =============================================================== panel b
BX = 262
panel_head(BX, "b", "Correctness × stability design")
CW, CH = 92, 84
G0X, G0Y = BX + 26, 44
cells = [  # col, row, fill, textfill, title rows, sub rows, key
    (0, 0, COR_L, INK, ["Correct,", "unstable"], ["right answer that", "varies with wording"], True),
    (1, 0, COR, "#ffffff", ["Correct,", "stable"], ["reliable parametric", "knowledge"], False),
    (0, 1, INC_L, INK, ["Incorrect,", "unstable"], ["wrong answer that", "varies with wording"], False),
    (1, 1, INC, "#ffffff", ["Incorrect,", "stable"], ["stably incorrect: the", "same error throughout"], True),
]
for col, row, fill, tf, title, sub, key in cells:
    x = G0X + col * (CW + 4); y = G0Y + row * (CH + 4)
    rect(x, y, CW, CH, fill=fill, stroke="none", rx=3)
    if key:
        rect(x - 1.5, y - 1.5, CW + 3, CH + 3, stroke=INK, sw=1.3, rx=4)
    lines(x + 8, y + 18, title, size=9.5, lh=11.5, w="700", fill=tf)
    lines(x + 8, y + 50, sub, size=7.6, lh=9.5, fill=tf if tf != INK else INK2)
# axes
gy2 = G0Y + 2 * CH + 4
o.append(f'<path d="M{G0X},{gy2 + 9} H{G0X + 2 * CW + 4} m-4,-3.5 l4,3.5 l-4,3.5" fill="none" '
         f'stroke="{INK}" stroke-width="0.75" stroke-linecap="round" stroke-linejoin="round"/>')
t(G0X + CW / 2, gy2 + 22, "low", size=8.5, fill=INK2, anchor="middle")
t(G0X + CW + 4 + CW / 2, gy2 + 22, "high", size=8.5, fill=INK2, anchor="middle")
t(G0X + CW + 2, gy2 + 35, "Stability (support, s)", size=9.5, w="700", anchor="middle")
ymid = G0Y + CH + 2
t(BX + 21, G0Y + CH / 2, "correct", size=8.5, fill=INK2, anchor="middle",
  extra=f' transform="rotate(-90 {BX + 21} {G0Y + CH / 2})"')
t(BX + 21, G0Y + CH + 4 + CH / 2, "incorrect", size=8.5, fill=INK2, anchor="middle",
  extra=f' transform="rotate(-90 {BX + 21} {G0Y + CH + 4 + CH / 2})"')
t(BX + 8, ymid, "Correctness (c)", size=9.5, w="700", anchor="middle",
  extra=f' transform="rotate(-90 {BX + 8} {ymid})"')
# notes
NY = gy2 + 56
rect(BX, NY - 9, 12, 9, stroke=INK, sw=1.3, rx=2)
lines(BX + 18, NY - 1, ["Off-diagonal cells separate correctness from",
                        "stability; contrasting only the diagonal cells",
                        "(stable-correct versus unstable-incorrect)",
                        "confounds the two."], size=8.5, lh=11, fill=INK2)
lines(BX, NY + 52, ["Stably incorrect facts (all six answers agree on",
                    "a wrong answer): 77–386 per model."], size=8.5, lh=11, fill=INK2)

flow_arrow(485, 150)

# =============================================================== panel c
CX = 504
CWD = W - CX - 1
panel_head(CX, "c", "Stability-controlled read-outs")
t(CX, 34, "Linear probe on h, cross-validated AUROC", size=8.5, fill=INK2)
t(CX, 45, "for correct versus incorrect", size=8.5, fill=INK2)

card_y = [56, 160, 264]
card_h = [96, 96, 96]
for y, h in zip(card_y, card_h):
    rect(CX, y, CWD, h, fill="#ffffff")

# -- card 1: matched pairs
y = card_y[0]
t(CX + 10, y + 15, "Stability-matched pairs", w="700")
t(CX + CWD - 10, y + 15, "Fig. 2a", size=8.5, fill=INK3, anchor="end")
ax0, ax1 = CX + 14, CX + CWD - 14
pts = [0.06, 0.2, 0.33, 0.47, 0.6, 0.74, 0.9]
for i, p in enumerate(pts):
    x = ax0 + p * (ax1 - ax0)
    dx = (-2.5, 2, -1.5, 2.5, -2, 1.5, -1)[i]
    line(x, y + 29, x + dx, y + 45, stroke=INK3, sw=0.6)
    circ(x, y + 29, 3.6, COR, stroke="#ffffff", sw=0.6)
    circ(x + dx, y + 45, 3.6, INC, stroke="#ffffff", sw=0.6)
o.append(f'<path d="M{ax0},{y + 55} H{ax1} m-4,-3 l4,3 l-4,3" fill="none" stroke="{INK}" '
         f'stroke-width="0.6" stroke-linecap="round" stroke-linejoin="round"/>')
t(ax0, y + 66, "support, s", size=8, fill=INK2)
t(ax1, y + 66, "pairs within 0.05", size=8, fill=INK2, anchor="end")
lines(CX + 10, y + 79, ["Each incorrect answer paired 1:1 with a correct one",
                        "of equal support, from the same dataset."], size=7.8, lh=9.5, fill=INK2)

# -- card 2: stability regimes
y = card_y[1]
t(CX + 10, y + 15, "Stability regimes", w="700")
t(CX + CWD - 10, y + 15, "Fig. 2b", size=8.5, fill=INK3, anchor="end")
seg = (ax1 - ax0 - 6) / 2
labels = [("answers disagree", True), ("all six answers agree", True)]
for i, (lab, used) in enumerate(labels):
    x = ax0 + i * (seg + 6)
    rect(x, y + 25, seg, 24, fill=PANEL if used else "#ffffff", stroke=INK if used else RULE,
         sw=0.9 if used else 0.6, rx=2, extra="" if used else ' stroke-dasharray="2 2"')
    if used:
        for k in range(3):
            circ(x + seg / 2 - 20 + k * 8, y + 37, 3.1, COR, stroke="#ffffff", sw=0.5)
            circ(x + seg / 2 - 20 + (k + 3) * 8, y + 37, 3.1, INC, stroke="#ffffff", sw=0.5)
    t(x + seg / 2, y + 60, lab, size=8, fill=INK2 if used else INK3, anchor="middle")
lines(CX + 10, y + 76, ["Matched pairs among unanimous facts compared",
                        "with matched pairs among all other facts."], size=7.8, lh=9.5, fill=INK2)

# -- card 3: scale
y = card_y[2]
t(CX + 10, y + 15, "Scale", w="700")
t(CX + CWD - 10, y + 15, "Fig. 2c–e", size=8.5, fill=INK3, anchor="end")
import math
sizes = [1, 1.7, 3, 3.8, 3.8, 4, 7, 8, 8, 12, 12, 14, 32]
xs = ax0 + 6
for i, s in enumerate(sizes):
    r = 1.6 + 1.25 * math.log2(s)
    xs += r
    circ(xs, y + 39, r, "#ffffff", stroke=INK, sw=0.8)
    xs += r + 2.6
t(ax0, y + 60, "1B", size=8, fill=INK2)
t(ax1, y + 60, "32B parameters", size=8, fill=INK2, anchor="end")
lines(CX + 10, y + 76, ["13 open-weight models from 5 families; each AUROC",
                        "regressed on log₂ parameter count."], size=7.8, lh=9.5, fill=INK2)

SVG = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="180mm" '
       f'height="{H / 4:.1f}mm" font-family="Arial, Helvetica, \'Liberation Sans\', sans-serif">\n'
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>\n' + "\n".join(o) + "\n</svg>\n")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out_dir", default=os.path.join(here, "..", "figures"))
    ap.add_argument("--name", default="figure1_design")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.abspath(os.path.join(args.out_dir, args.name))
    with open(base + ".svg", "w", encoding="utf-8") as f:
        f.write(SVG)
    print(f"saved {base}.svg")
    chrome = next((c for c in ("google-chrome", "chromium", "chromium-browser") if shutil.which(c)), None)
    if not chrome:
        print("  (no Chrome found: open the SVG in Inkscape/Illustrator to export PDF/PNG)")
        return
    html = base + "_tmp.html"
    with open(html, "w", encoding="utf-8") as f:
        f.write(f'<html><head><style>@page{{size:180mm {H / 4:.1f}mm;margin:0}}'
                f'html,body{{margin:0;background:#fff}}svg{{display:block}}</style></head>'
                f'<body>{SVG}</body></html>')
    common = [chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars"]
    subprocess.run(common + ["--no-pdf-header-footer", f"--print-to-pdf={base}.pdf", "file://" + html],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(common + ["--force-device-scale-factor=6", f"--window-size={int(W * 0.9449) + 1},{int(H * 0.9449) + 1}",
                             f"--screenshot={base}.png", "file://" + html],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(html)
    print(f"saved {base}.pdf / .png")


if __name__ == "__main__":
    main()
