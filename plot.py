"""Plot stratified eval losses from eval_log.jsonl using Plotly."""

import json
import plotly.graph_objects as go  # type: ignore[import-untyped]


def load_eval_log(path: str = "eval_log.jsonl") -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot(records: list[dict]) -> None:
    steps = [r["step"] for r in records]

    loss_keys = [k for k in records[0] if k.endswith("_loss")]
    # Sort so eval_loss comes first, then mask buckets in order
    loss_keys.sort(key=lambda k: (0 if k == "eval_loss" else int(k.split("_")[0].replace("mask", ""))))

    colors = {
        "eval_loss": "#888888",
        "mask10_loss": "#22c55e",
        "mask30_loss": "#3b82f6",
        "mask50_loss": "#a855f7",
        "mask70_loss": "#f97316",
        "mask90_loss": "#ef4444",
    }

    labels = {
        "eval_loss": "eval (mixed)",
        "mask10_loss": "10% masked",
        "mask30_loss": "30% masked",
        "mask50_loss": "50% masked",
        "mask70_loss": "70% masked",
        "mask90_loss": "90% masked",
    }

    fig = go.Figure()

    for key in loss_keys:
        values = [r.get(key) for r in records]
        fig.add_trace(go.Scatter(
            x=steps,
            y=values,
            name=labels.get(key, key),
            mode="lines+markers",
            line=dict(color=colors.get(key, "#888"), width=2),
            marker=dict(size=6),
        ))

    fig.update_layout(
        title="Stratified Eval Loss by Masking Rate",
        xaxis_title="Training Step",
        yaxis_title="Loss",
        template="plotly_dark",
        font=dict(size=14),
        legend=dict(x=1.02, y=1, bordercolor="#444", borderwidth=1),
        hovermode="x unified",
        width=1000,
        height=600,
    )

    fig.show()
    print("Plot opened in browser.")


if __name__ == "__main__":
    records = load_eval_log()
    plot(records)
