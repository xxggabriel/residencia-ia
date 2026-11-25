"""Shared, physically sized publication style and paired notebook/Docs exports."""

from __future__ import annotations

import hashlib
import json
import os
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ft2font import FT2Font
from matplotlib.text import Text

INK = "#163247"
MUTED = "#496373"
GRID = "#DDE7EC"
BLUE = "#176B91"
TEAL = "#087F78"
ORANGE = "#C55D32"
PALETTE = ["#718390", BLUE, ORANGE, "#8369A4", TEAL, "#A55A83", "#C39327", "#3D85A7", "#324D67"]


POLICY_COLORS = {
    "frozen": "#718390",
    "cara": "#176B91",
    "upf": "#C55D32",
    "oracle": "#8369A4",
    "drift_adwin": "#087F78",
    "adwin": "#087F78",
    "online": "#A55A83",
    "incremental": "#108F91",
    "periodic": "#C39327",
    "drift": "#335671",
    "drift_ddm": "#608AAB",
    "ddm": "#608AAB",
    "always_retrain": "#496373",
    "oracle_staleness": "#7D594E",
    "markov": "#8A762B",
}


def apply_style():
    if matplotlib.get_backend().lower() != "agg":
        plt.switch_backend("Agg")
    supplied = os.getenv("BENCHMARK_ARIAL_FONT")
    if supplied:
        font_manager.fontManager.addfont(Path(supplied).expanduser())
    try:
        path = Path(font_manager.findfont("Arial", fallback_to_default=False)).resolve()
    except ValueError as error:
        raise ValueError("Arial é obrigatória. Configure BENCHMARK_ARIAL_FONT se necessário.") from error
    if FT2Font(str(path)).family_name != "Arial":
        raise ValueError("A fonte resolvida não é Arial.")
    matplotlib.rcdefaults()
    matplotlib.rcParams.update(
        {
            "font.family": ["Arial"],
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.usetex": False,
        }
    )
    return {
        "family": "Arial",
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "minimum_points": 11,
        "fallback": False,
    }


def decorate(
    fig,
    title,
    caption,
    *,
    show_title=True,
    show_footer=True,
    footer=None,
    metric_descriptions=None,
    **metadata,
):
    """Keep the full caption external; the optional image footer is deliberately short."""
    if show_title:
        fig.suptitle(
            textwrap.fill(title, width=51),
            x=0.025,
            y=0.985,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            color=INK,
        )
    if show_footer:
        brief = footer or "Resultados observados. Consulte a legenda para o protocolo e os denominadores."
        lines = textwrap.wrap(brief, width=79)
        if len(lines) > 2:
            raise ValueError("O rodapé interno deve ter no máximo duas linhas legíveis.")
        fig.text(0.025, 0.012, "\n".join(lines), ha="left", va="bottom", fontsize=11, color=MUTED)
    fig._benchmark_metadata = {
        "title": title,
        "caption": caption,
        "show_title": bool(show_title),
        "show_footer": bool(show_footer),
        "metric_descriptions": metric_descriptions or {},
        **metadata,
    }
    return fig


def validate_figure(fig):
    width_inches, height_inches = fig.get_size_inches()
    if abs(width_inches - 6.2) > 1e-8 or height_inches > 6.5 + 1e-8:
        raise ValueError("A figura deve ter 6,2 polegadas de largura e até 6,5 de altura.")
    if not isinstance(fig.canvas, FigureCanvasAgg):
        FigureCanvasAgg(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    records = []
    excluded_tick_labels = set()
    # Matplotlib creates labels for locator ticks outside the current view and
    # deliberately omits them when drawing. Inspect only the ticks actually drawn.
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            for tick in (*axis.get_major_ticks(), *axis.get_minor_ticks()):
                if not low - 1e-9 <= tick.get_loc() <= high + 1e-9:
                    excluded_tick_labels.update((id(tick.label1), id(tick.label2)))
    for text in fig.findobj(match=Text):
        if id(text) in excluded_tick_labels or not text.get_visible() or not text.get_text():
            continue
        face = Path(font_manager.findfont(text.get_fontproperties(), fallback_to_default=False))
        if text.get_fontsize() < 11 or FT2Font(str(face)).family_name != "Arial":
            raise ValueError("Todo texto visível deve usar Arial com pelo menos 11 pontos.")
        bounds = text.get_window_extent(renderer)
        if min(bounds.x0, bounds.y0) < -1 or bounds.x1 > width + 1 or bounds.y1 > height + 1:
            raise ValueError("Texto cortado na figura: " + text.get_text())
        records.append(
            {
                "text": text.get_text(),
                "font_points": text.get_fontsize(),
                "bounds": [float(v) for v in (bounds.x0, bounds.y0, bounds.x1, bounds.y1)],
            }
        )
    return {
        "width_inches": float(width_inches),
        "height_inches": float(height_inches),
        "minimum_font_points": 11,
        "all_text_inside_canvas": True,
        "text": records,
    }


def export_bundle(figures, output, variant="notebook"):
    """Export editable SVG, embedded-font PDF and 300-dpi PNG without changing the physical size."""
    root = Path(output) / variant
    folder = root / "figuras"
    folder.mkdir(parents=True, exist_ok=True)
    pdf_path = root / "figuras-benchmarks.pdf"
    captions, layouts = {}, {}
    generated = [pdf_path]
    with PdfPages(pdf_path) as pdf:
        for name, fig in figures.items():
            if variant == "docs":
                meta = getattr(fig, "_benchmark_metadata", {})
                if meta.get("show_title") is not False or meta.get("show_footer") is not False:
                    raise ValueError("A versão Docs exige show_title=False e show_footer=False.")
            layouts[name] = validate_figure(fig)
            for extension in ("png", "pdf", "svg"):
                asset = folder / f"{name}.{extension}"
                fig.savefig(asset, dpi=300)
                generated.append(asset)
            pdf.savefig(fig)
            captions[name] = getattr(fig, "_benchmark_metadata", {})
    (root / "legendas.json").write_text(json.dumps(captions, ensure_ascii=False, indent=2) + "\n")
    markdown = []
    for name, item in captions.items():
        markdown.extend(["## " + item.get("title", name), "", item.get("caption", ""), ""])
        for metric, meaning in item.get("metric_descriptions", {}).items():
            markdown.extend([f"- **{metric}:** {meaning}"])
        markdown.append("")
    (root / "legendas.md").write_text("\n".join(markdown))
    (root / "layout.json").write_text(json.dumps(layouts, ensure_ascii=False, indent=2) + "\n")
    manifest = [
        {
            "path": p.relative_to(root).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(generated + [root / "legendas.json", root / "legendas.md", root / "layout.json"])
    ]
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return pdf_path
