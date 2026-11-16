"""Dependency-free SVG charts: multi-series lines with event markers, bars and heatmaps.

Output is deterministic for identical data, renders in the MLflow artifact viewer
and stays vector-quality for papers. Missing values (None) break a line.
"""

import math
from xml.sax.saxutils import escape

PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
    "#999999",
    "#882255",
)
MARKER_STYLE = {
    "alarm": ("#D55E00", "4 3"),
    "changepoint": ("#000000", "8 4"),
    "update": ("#009E73", "2 3"),
    "rejected": ("#999999", "2 3"),
    "bound": ("#CC79A7", "6 3"),
}
WIDTH, HEIGHT = 960, 440
LEFT, RIGHT, TOP, BOTTOM = 78, 270, 48, 58


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e5 or magnitude < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4g}"


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if not math.isfinite(low) or not math.isfinite(high):
        return [0.0]
    if high == low:
        low, high = low - 0.5, high + 0.5
    raw = (high - low) / max(count, 1)
    power = 10 ** math.floor(math.log10(raw))
    step = next(m * power for m in (1, 2, 2.5, 5, 10) if m * power >= raw)
    start = math.floor(low / step) * step
    ticks, value = [], start
    while value <= high + step * 1e-9:
        ticks.append(round(value, 12))
        value += step
    return ticks


def _text(x, y, content, *, size=12, anchor="start", weight="normal", rotate=None, fill="#222222"):
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="{fill}"{transform}>{escape(str(content))}</text>'
    )


def _document(body: list[str], title: str, width=WIDTH, height=HEIGHT) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">',
            f"<title>{escape(title)}</title>",
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
            _text(LEFT, 28, title, size=16, weight="bold"),
            *body,
            "</svg>\n",
        ]
    )


def _empty(title: str, reason: str) -> str:
    return _document([_text(WIDTH / 2, HEIGHT / 2, reason, size=14, anchor="middle", fill="#777777")], title)


def line_chart(title, series, *, x_label="", y_label="", markers=(), y_range=None) -> str:
    """series: [{"name", "points": [[x, y|None], ...]}]; markers: [{"x", "kind", "label"}]."""
    values = [(x, y) for s in series for x, y in s["points"] if y is not None and math.isfinite(y)]
    if not values:
        return _empty(title, "no finite values recorded")
    xs = [x for x, _ in values] + [m["x"] for m in markers]
    x_low, x_high = min(xs), max(xs)
    if x_high == x_low:
        x_low, x_high = x_low - 1, x_high + 1
    y_low, y_high = y_range or (min(y for _, y in values), max(y for _, y in values))
    y_ticks = nice_ticks(y_low, y_high)
    y_low, y_high = min(y_ticks[0], y_low), max(y_ticks[-1], y_high)
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

    def px(x):
        return LEFT + (x - x_low) / (x_high - x_low) * plot_w

    def py(y):
        return TOP + plot_h - (y - y_low) / ((y_high - y_low) or 1) * plot_h

    body = []
    for tick in y_ticks:
        y = py(tick)
        body.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{LEFT + plot_w}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        body.append(_text(LEFT - 8, y + 4, _fmt(tick), size=11, anchor="end"))
    for tick in nice_ticks(x_low, x_high, 8):
        if x_low <= tick <= x_high:
            x = px(tick)
            body.append(
                f'<line x1="{x:.1f}" y1="{TOP + plot_h}" x2="{x:.1f}" y2="{TOP + plot_h + 5}" '
                f'stroke="#555555"/>'
            )
            body.append(_text(x, TOP + plot_h + 19, _fmt(tick), size=11, anchor="middle"))
    body.append(
        f'<rect x="{LEFT}" y="{TOP}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#555555"/>'
    )
    for marker in markers:
        color, dash = MARKER_STYLE.get(marker.get("kind"), ("#777777", "3 3"))
        x = px(marker["x"])
        body.append(
            f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{TOP + plot_h}" stroke="{color}" '
            f'stroke-dasharray="{dash}" stroke-width="1"><title>{escape(marker.get("label", ""))}'
            "</title></line>"
        )
    for index, entry in enumerate(series):
        color = PALETTE[index % len(PALETTE)]
        segment = []
        segments = []
        for x, y in entry["points"]:
            if y is None or not math.isfinite(y):
                if segment:
                    segments.append(segment)
                segment = []
            else:
                segment.append((px(x), py(y)))
        if segment:
            segments.append(segment)
        for points in segments:
            if len(points) == 1:
                body.append(f'<circle cx="{points[0][0]:.1f}" cy="{points[0][1]:.1f}" r="3" fill="{color}"/>')
            else:
                path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                body.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="1.8"/>')
    legend_x, legend_y = LEFT + plot_w + 16, TOP + 6
    entries = [(entry["name"], PALETTE[i % len(PALETTE)], None) for i, entry in enumerate(series)]
    kinds = sorted({m.get("kind") for m in markers})
    entries += [(kind, *MARKER_STYLE.get(kind, ("#777777", "3 3"))) for kind in kinds]
    for offset, (name, color, dash) in enumerate(entries[:24]):
        y = legend_y + offset * 17
        style = f' stroke-dasharray="{dash}"' if dash else ' stroke-width="2.5"'
        body.append(
            f'<line x1="{legend_x}" y1="{y:.1f}" x2="{legend_x + 18}" y2="{y:.1f}" stroke="{color}"{style}/>'
        )
        body.append(_text(legend_x + 24, y + 4, name, size=10))
    body.append(_text(LEFT + plot_w / 2, HEIGHT - 14, x_label, size=12, anchor="middle"))
    body.append(_text(20, TOP + plot_h / 2, y_label, size=12, anchor="middle", rotate=-90))
    return _document(body, title)


def bar_chart(title, labels, values, *, x_label="") -> str:
    """Horizontal bars; None values are shown as 'unavailable'."""
    if not labels:
        return _empty(title, "no values recorded")
    finite = [v for v in values if v is not None and math.isfinite(v)]
    low, high = min([0.0, *finite]), max([0.0, *finite]) if finite else 1.0
    ticks = nice_ticks(low, high)
    low, high = min(ticks[0], low), max(ticks[-1], high)
    row, label_w = 22, 260
    height = TOP + BOTTOM + row * len(labels)
    plot_w = WIDTH - label_w - 60

    def px(v):
        return label_w + (v - low) / ((high - low) or 1) * plot_w

    body = []
    for tick in ticks:
        x = px(tick)
        body.append(f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{height - BOTTOM}" stroke="#e6e6e6"/>')
        body.append(_text(x, height - BOTTOM + 18, _fmt(tick), size=11, anchor="middle"))
    for index, (label, value) in enumerate(zip(labels, values)):
        y = TOP + index * row
        body.append(_text(label_w - 8, y + row * 0.68, label, size=11, anchor="end"))
        if value is None or not math.isfinite(value):
            body.append(_text(px(0) + 4, y + row * 0.68, "unavailable", size=11, fill="#999999"))
            continue
        start, end = sorted((px(0), px(value)))
        color = PALETTE[0] if value >= 0 else PALETTE[1]
        body.append(
            f'<rect x="{start:.1f}" y="{y + 3}" width="{max(end - start, 1):.1f}" height="{row - 6}" '
            f'fill="{color}"><title>{escape(label)}: {_fmt(value)}</title></rect>'
        )
        body.append(_text(end + 4, y + row * 0.68, _fmt(value), size=10, fill="#444444"))
    body.append(_text(label_w + plot_w / 2, height - 14, x_label, size=12, anchor="middle"))
    return _document(body, title, height=height)


def heatmap(
    title, row_labels, column_labels, matrix, *, x_label="", y_label="", value_range=(0.0, 1.0)
) -> str:
    if not row_labels or not column_labels:
        return _empty(title, "no completed rows")
    cell = max(28, min(70, int((WIDTH - LEFT - RIGHT) / len(column_labels))))
    width = LEFT + cell * len(column_labels) + 120
    height = TOP + cell * len(row_labels) + BOTTOM
    low, high = value_range
    body = []
    for r, row in enumerate(matrix):
        body.append(_text(LEFT - 8, TOP + r * cell + cell / 2 + 4, row_labels[r], size=11, anchor="end"))
        for c, value in enumerate(row):
            x, y = LEFT + c * cell, TOP + r * cell
            if value is None:
                fill, label = "#f2f2f2", "–"
            else:
                share = min(max((value - low) / ((high - low) or 1), 0.0), 1.0)
                fill = f"rgb({int(247 - 200 * share)},{int(251 - 120 * share)},{int(255 - 75 * share)})"
                label = f"{value:.2f}"
            body.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff"/>'
            )
            body.append(_text(x + cell / 2, y + cell / 2 + 4, label, size=10, anchor="middle"))
    for c, label in enumerate(column_labels):
        body.append(
            _text(
                LEFT + c * cell + cell / 2, TOP + len(row_labels) * cell + 16, label, size=11, anchor="middle"
            )
        )
    body.append(_text(LEFT + cell * len(column_labels) / 2, height - 12, x_label, size=12, anchor="middle"))
    body.append(_text(20, TOP + cell * len(row_labels) / 2, y_label, size=12, anchor="middle", rotate=-90))
    return _document(body, title, width=width, height=height)
