import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# =========================
# Глобальные настройки шрифта
# =========================
FONT_FAMILY = "Times New Roman"   # если не установлен, matplotlib подменит
FONT_FAMILY_FALLBACKS = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]

FONT_BASE = 28
FONT_LAYER_TITLE = 28
FONT_OPS = 28
FONT_OPS_GAUSS = 28

ARROW_LW = 1.8
TICK_LW = 1.6

def draw_stack(ax, x, y, w, h, n=6, dx=0.18, dy=0.12,
               face="#bfbfbf", edge="#666666", alpha=1.0):
    for i in range(n):
        ax.add_patch(Rectangle(
            (x + i*dx, y + i*dy), w, h,
            facecolor=face, edgecolor=edge, linewidth=1.0, alpha=alpha
        ))
    return (x + (n-1)*dx, y + (n-1)*dy, w, h)

def label(ax, x, y, s, size=FONT_BASE, weight="normal", ha="center", va="bottom"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va)

def arrow(ax, xy1, xy2, lw=ARROW_LW):
    ax.add_patch(FancyArrowPatch(
        xy1, xy2, arrowstyle="-|>", mutation_scale=14,
        linewidth=lw, color="black"
    ))

def draw_lenet_like_original_ops_fixed(
    save_png="lenet_like_original_ops_fixed.png",
    save_svg="lenet_like_original_ops_fixed.svg"
):
    # Шрифты: serif + попытка использовать Times New Roman
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": FONT_FAMILY_FALLBACKS,
        "axes.unicode_minus": False,
        "font.size": FONT_BASE,
        "font.weight": "normal",  # глобально без жирного
    })

    fig, ax = plt.subplots(figsize=(19, 5.0))
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # ---- Layers ----
    inp = draw_stack(ax, x=0.9, y=2.35, w=2.2, h=2.2, n=1, dx=0, dy=0, face="#d9d9d9")
    label(ax, 2.0, 6.55, "INPUT\n32×32", size=FONT_LAYER_TITLE, weight="normal")

    # Буква A во входе
    ix, iy, iw, ih = inp
    A_left  = (ix + 0.30*iw, iy + 0.20*ih)
    A_top   = (ix + 0.55*iw, iy + 0.85*ih)
    A_right = (ix + 0.80*iw, iy + 0.20*ih)

    ax.plot([A_left[0], A_top[0]],  [A_left[1], A_top[1]],
            color="black", lw=5, solid_capstyle="round")
    ax.plot([A_top[0], A_right[0]], [A_top[1], A_right[1]],
            color="black", lw=5, solid_capstyle="round")

    cross_y = iy + 0.52*ih
    ax.plot([ix + 0.40*iw, ix + 0.72*iw], [cross_y, cross_y],
            color="black", lw=5, solid_capstyle="round")

    c1 = draw_stack(ax, x=4.7, y=2.10, w=2.0, h=2.0, n=6, dx=0.16, dy=0.10, face="#bfbfbf")
    label(ax, 5.7, 6.55, "C1: feature maps\n6@28×28", size=FONT_LAYER_TITLE, weight="normal")

    s2 = draw_stack(ax, x=8.8, y=2.45, w=1.6, h=1.6, n=6, dx=0.14, dy=0.09, face="#c9c9c9")
    label(ax, 9.7, 6.55, "S2: f. maps\n6@14×14", size=FONT_LAYER_TITLE, weight="normal")

    c3 = draw_stack(ax, x=12.7, y=2.25, w=1.7, h=1.7, n=16, dx=0.08, dy=0.055, face="#bdbdbd")
    label(ax, 13.6, 6.55, "C3: f. maps\n16@10×10", size=FONT_LAYER_TITLE, weight="normal")

    s4 = draw_stack(ax, x=17.0, y=2.60, w=1.3, h=1.3, n=16, dx=0.065, dy=0.045, face="#c7c7c7")
    label(ax, 17.75, 6.55, "S4: f. maps\n16@5×5", size=FONT_LAYER_TITLE, weight="normal")

    # ---- Fully connected blocks ----
    fc_y = 2.75
    c5_x = 20.6
    gap  = 2.3
    w_fc = 1.2
    h_fc = 0.55

    ax.add_patch(Rectangle((c5_x, fc_y), w_fc, h_fc, facecolor="#b3b3b3", edgecolor="#666666", lw=1.0))
    f6_x = c5_x + w_fc + gap
    ax.add_patch(Rectangle((f6_x, fc_y), w_fc, h_fc, facecolor="#ffffff", edgecolor="#666666", lw=1.0))
    out_x = f6_x + w_fc + gap
    ax.add_patch(Rectangle((out_x, fc_y), w_fc, h_fc, facecolor="#b3b3b3", edgecolor="#666666", lw=1.0))

    right_edge = out_x + w_fc + 1.2
    ax.set_xlim(0, right_edge)

    label(ax, c5_x, 6.00, "C5: layer\n120", size=FONT_LAYER_TITLE, weight="normal", ha="left")
    label(ax, f6_x, 5.35, "F6: layer\n84",  size=FONT_LAYER_TITLE, weight="normal", ha="left")
    label(ax, out_x, 4.70, "OUTPUT\n10",    size=FONT_LAYER_TITLE, weight="normal", ha="left")

    # ---- Arrow helpers ----
    def right_center(b): return (b[0] + b[2], b[1] + b[3]/2)
    def left_center(b):  return (b[0],        b[1] + b[3]/2)

    # ---- ONLY transition arrows ----
    arrow(ax, right_center(inp), left_center(c1))
    arrow(ax, right_center(c1),  left_center(s2))
    arrow(ax, right_center(s2),  left_center(c3))
    arrow(ax, right_center(c3),  left_center(s4))
    arrow(ax, right_center(s4), (c5_x, fc_y + h_fc/2))
    arrow(ax, (c5_x + w_fc, fc_y + h_fc/2), (f6_x, fc_y + h_fc/2))
    arrow(ax, (f6_x + w_fc, fc_y + h_fc/2), (out_x, fc_y + h_fc/2))

    # ---- Operation labels ----
    y_ops = 0.72
    y_tick0, y_tick1 = 1.15, 1.45

    def op_mark(x, text, size=FONT_OPS):
        ax.plot([x, x], [y_tick0, y_tick1], color="black", lw=TICK_LW)
        label(ax, x, y_ops, text, size=size, weight="normal", va="top")

    def midx(a, b): return (a + b) / 2

    op_mark(midx(right_center(inp)[0], left_center(c1)[0]), "Convolutions")
    op_mark(midx(right_center(c1)[0],  left_center(s2)[0]), "Subsampling")
    op_mark(midx(right_center(s2)[0],  left_center(c3)[0]), "Convolutions")
    op_mark(midx(right_center(c3)[0],  left_center(s4)[0]), "Subsampling")

    op_mark(midx(right_center(s4)[0],  c5_x),  "Full\nconnection")
    op_mark(midx(c5_x + w_fc,          f6_x),  "Full\nconnection")
    op_mark(midx(f6_x + w_fc,          out_x), "Gaussian\nconnections", size=FONT_OPS_GAUSS)

    fig.tight_layout(pad=0.2)
    fig.savefig(save_png, dpi=600, bbox_inches="tight")
    fig.savefig(save_svg, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    draw_lenet_like_original_ops_fixed()
    print("Готово: lenet_like_original_ops_fixed.png (600 dpi) и lenet_like_original_ops_fixed.svg")
