from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def generate_comparison_chart(
    data_json_path: Path,
    output_root: Path,
    title: str = "Trained vs Untrained DistilBERT",
    subtitle: str = "Comparison across supplied datasets",
) -> dict[str, str]:
    with data_json_path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    datasets = _extract_datasets(raw_data)
    if not datasets:
        raise ValueError("No valid dataset entries were found in the input JSON.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    svg_content = _build_svg_chart(datasets, title=title, subtitle=subtitle)

    chart_path = run_dir / "comparison_chart.svg"
    normalized_path = run_dir / "normalized_chart_data.json"
    chart_path.write_text(svg_content, encoding="utf-8")
    normalized_path.write_text(json.dumps({"datasets": datasets}, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "chart_path": str(chart_path),
        "data_path": str(normalized_path),
    }


def _extract_datasets(raw_data: dict[str, Any]) -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    for key, value in raw_data.items():
        if key == "created_at":
            continue
        if not isinstance(value, dict):
            continue
        trained = value.get("trained")
        untrained = value.get("untrained")
        dataset_label = value.get("dataset", key)
        if not isinstance(trained, dict) or not isinstance(untrained, dict):
            continue
        if "accuracy" not in trained or "f1" not in trained or "accuracy" not in untrained or "f1" not in untrained:
            continue
        datasets.append(
            {
                "key": key,
                "label": dataset_label,
                "trained": {
                    "accuracy": float(trained["accuracy"]),
                    "f1": float(trained["f1"]),
                },
                "untrained": {
                    "accuracy": float(untrained["accuracy"]),
                    "f1": float(untrained["f1"]),
                },
            }
        )
    return datasets


def _build_svg_chart(datasets: list[dict[str, Any]], title: str, subtitle: str) -> str:
    width, height = 1100, 520
    margin = 70
    panel_gap = 70
    panel_width = (width - margin * 2 - panel_gap) // 2
    panel_height = 320
    baseline = 420
    metrics = [("accuracy", "Accuracy"), ("f1", "F1")]
    colors = {"Trained": "#2b6cb0", "Untrained": "#9aa5b1"}

    bars: list[tuple[str, str, dict[str, float]]] = []
    for dataset in datasets:
        bars.append((dataset["label"], "Trained", dataset["trained"]))
        bars.append((dataset["label"], "Untrained", dataset["untrained"]))

    svg: list[str] = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"]
    svg.append("<rect width='100%' height='100%' fill='white'/>")
    svg.append(f"<text x='550' y='36' text-anchor='middle' font-family='Segoe UI, Arial' font-size='24' font-weight='700'>{title}</text>")
    svg.append(f"<text x='550' y='60' text-anchor='middle' font-family='Segoe UI, Arial' font-size='14' fill='#4a5568'>{subtitle}</text>")

    for panel_index, (metric_key, metric_label) in enumerate(metrics):
        x0 = margin + panel_index * (panel_width + panel_gap)
        svg.append(f"<text x='{x0 + panel_width / 2}' y='100' text-anchor='middle' font-family='Segoe UI, Arial' font-size='18' font-weight='600'>{metric_label}</text>")
        for tick in range(6):
            y = baseline - (panel_height * tick / 5)
            value = tick / 5
            svg.append(f"<line x1='{x0}' y1='{y}' x2='{x0 + panel_width}' y2='{y}' stroke='#e2e8f0' stroke-width='1'/>")
            svg.append(f"<text x='{x0 - 10}' y='{y + 5}' text-anchor='end' font-family='Segoe UI, Arial' font-size='12' fill='#718096'>{value:.1f}</text>")

        bar_width = 56
        gap = 26
        start = x0 + 40
        for index, (dataset_label, status, values) in enumerate(bars):
            x = start + index * (bar_width + gap)
            value = values[metric_key]
            bar_height = panel_height * value
            y = baseline - bar_height
            svg.append(f"<rect x='{x}' y='{y}' width='{bar_width}' height='{bar_height}' rx='6' fill='{colors[status]}'/>")
            svg.append(f"<text x='{x + bar_width / 2}' y='{y - 8}' text-anchor='middle' font-family='Segoe UI, Arial' font-size='12' fill='#1a202c'>{value:.2f}</text>")
            svg.append(f"<text x='{x + bar_width / 2}' y='{baseline + 22}' text-anchor='middle' font-family='Segoe UI, Arial' font-size='11' fill='#2d3748'>{dataset_label}</text>")
            svg.append(f"<text x='{x + bar_width / 2}' y='{baseline + 38}' text-anchor='middle' font-family='Segoe UI, Arial' font-size='11' fill='#718096'>{status}</text>")

    legend_y = 480
    legend_x = 410
    for index, status in enumerate(["Trained", "Untrained"]):
        x = legend_x + index * 150
        svg.append(f"<rect x='{x}' y='{legend_y - 12}' width='16' height='16' fill='{colors[status]}' rx='3'/>")
        svg.append(f"<text x='{x + 24}' y='{legend_y + 1}' font-family='Segoe UI, Arial' font-size='13' fill='#2d3748'>{status}</text>")

    svg.append("</svg>")
    return "\n".join(svg)
