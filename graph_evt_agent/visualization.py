"""Matplotlib figures for detection, labelling, training quality and evaluation.

Install the optional extra with ``pip install 'graph-evt-agent[viz]'``. Every
function returns the ``matplotlib.figure.Figure`` so callers can save it
(``fig.savefig("training.png", dpi=150)``) or embed it in a notebook.
Colors keep a fixed meaning: heuristic = aqua, GNN = blue, GAT = orange,
1-D route = violet, n-D route = pink; training curves are solid and
validation curves dashed.
"""

from importlib import import_module
from typing import Any, Sequence

import numpy as np

COLORS = {"heuristic": "#1baf7a", "gnn": "#2a78d6", "gat": "#eb6834"}
LABELS = {"heuristic": "Heuristic ranker", "gnn": "GNN", "gat": "GAT"}
ALARM = "#e34948"
MUTED = "#52514e"
NEUTRAL = "#b8b7b1"
ROUTE_COLORS = {"1-d": "#4a3aa7", "n-d": "#e87ba4"}
# Pseudo-labels come from the heuristic ranker, so they share its color.
ORIGIN_COLORS = {"manual": "#4a3aa7", "pseudo": COLORS["heuristic"], "abstained": NEUTRAL}


def _pyplot() -> Any:
    try:
        return import_module("matplotlib.pyplot")
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError("plotting requires matplotlib: pip install "
                          "'graph-evt-agent[viz]'") from error


def _style(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)


def plot_detection(result: Any, timestamps: Sequence[float] | None = None,
                   channel_names: Sequence[str] | None = None) -> Any:
    """Indicator vs. POT/GPD threshold and, for n-D, the source ranking."""
    plt = _pyplot()
    detection = result.detection
    names = list(channel_names or result.input_profile.channel_names)
    time = np.arange(len(detection.indicator)) if timestamps is None else np.asarray(timestamps)
    panels = 2 if result.ranking is not None else 1
    fig, axes = plt.subplots(panels, 1, figsize=(11, 3.4 * panels), constrained_layout=True,
                             squeeze=False)
    ax = axes[0, 0]
    ax.plot(time, detection.indicator, color=COLORS["gnn"], linewidth=1.2, label="EVT indicator")
    ax.axhline(detection.alarm_threshold, color=ALARM, linestyle="--", linewidth=1.5,
               label="POT/GPD alarm threshold")
    if detection.time_index is not None:
        ax.axvline(time[detection.time_index], color=MUTED, linewidth=1.5, label="first alarm")
    ax.set(title=f"EVT detection ({result.input_profile.kind})", ylabel="robust top-k score")
    ax.legend(loc="upper left", frameon=False)
    _style(ax)
    if result.ranking is not None:
        ax = axes[1, 0]
        weights = np.empty(len(names))
        weights[result.ranking.node_ids] = result.ranking.probabilities
        colors = [COLORS["gnn"] if node == result.ranking.source else NEUTRAL
                  for node in range(len(names))]
        ax.bar(names, weights, color=colors, edgecolor="white", linewidth=2)
        ax.set(title="Transparent source ranking (highlighted = top source)",
               ylabel="relative weight", ylim=(0, 1))
        _style(ax)
    return fig


def plot_labelled_recording(values: np.ndarray, labels: Any, series_id: str | None = None,
                            channel_names: Sequence[str] | None = None,
                            baseline_size: int | None = None) -> Any:
    """Stacked channels with every detected event marked by label origin."""
    plt = _pyplot()
    data = np.asarray(values, dtype=float)
    data = data[:, None] if data.ndim == 1 else data
    episodes = [item for item in labels.episodes
                if series_id is None or item.series_id == series_id]
    names = list(channel_names or (episodes[0].channel_names if episodes else
                                   [f"channel_{i}" for i in range(data.shape[1])]))
    spread = 1.2 * max(float(np.nanmax(np.abs(data - np.nanmedian(data, axis=0)))), 1e-9)
    fig, ax = plt.subplots(figsize=(12, 0.7 * data.shape[1] + 2.5), constrained_layout=True)
    for node in range(data.shape[1]):
        offset = -node * spread
        ax.plot(data[:, node] - np.median(data[:, node]) + offset, color=MUTED, linewidth=0.6)
        ax.text(-0.01 * len(data), offset, names[node], ha="right", va="center", fontsize=9)
    if baseline_size:
        ax.axvspan(0, baseline_size, color="#e6e5e0", alpha=0.6, label="baseline")
    seen = set()
    for item in episodes:
        color = ORIGIN_COLORS[item.origin]
        label = None if item.origin in seen else f"{item.origin} label"
        seen.add(item.origin)
        ax.axvline(item.event_time, color=color, linewidth=1.6, label=label)
        if item.source is not None:
            ax.scatter([item.event_time], [-item.source * spread], s=70, color=color,
                       edgecolor="white", linewidth=2, zorder=3)
            ax.annotate(f"{item.source_name}\n{item.confidence:.2f}",
                        (item.event_time, spread * 0.5), fontsize=8, color=MUTED,
                        ha="center", va="bottom")
    ax.set_ylim(-(data.shape[1] - 0.4) * spread, 1.1 * spread)
    ax.set(title="Detected episodes and source labels (dot = labelled source, weight)",
           xlabel="sample", yticks=[])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    return fig


def plot_training_history(report_or_history: Any, axes: Sequence[Any] | None = None) -> Any:
    """Loss and top-1 accuracy per epoch for GNN and GAT (train solid, val dashed)."""
    plt = _pyplot()
    history = getattr(report_or_history, "history", report_or_history)
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    else:
        fig = axes[0].figure
    epochs = history["epoch"]
    best = getattr(report_or_history, "best_epoch", None)
    for ax, metric, title in ((axes[0], "loss", "Cross-entropy loss"),
                              (axes[1], "accuracy", "Top-1 accuracy")):
        for model in ("gnn", "gat"):
            for split, style in (("train", "-"), ("val", "--")):
                series = np.asarray(history[f"{model}_{split}_{metric}"], dtype=float)
                if np.isfinite(series).any():
                    ax.plot(epochs, series, style, color=COLORS[model], linewidth=2,
                            label=f"{LABELS[model]} {'train' if split == 'train' else 'validation'}")
        if best is not None:
            ax.axvline(best, color=MUTED, linestyle=":", linewidth=1.5,
                       label=f"best validation epoch ({best})")
        ax.set(title=title, xlabel="epoch")
        if metric == "accuracy":
            ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False)
        _style(ax)
    return fig


def plot_model_comparison(report: Any, split: str = "validation", ax: Any = None) -> Any:
    """Grouped bars of Top-1, Top-k and MRR for heuristic, GNN and GAT."""
    plt = _pyplot()
    metrics = report.validation_metrics if split == "validation" else report.train_metrics
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    else:
        fig = ax.figure
    methods = [method for method in ("heuristic", "gnn", "gat") if method in metrics]
    k = metrics[methods[0]].k if methods else 3
    groups = ["top1", "topk", "mrr"]
    width = 0.8 / max(len(methods), 1)
    for index, method in enumerate(methods):
        heights = [getattr(metrics[method], name) for name in groups]
        positions = np.arange(len(groups)) + (index - (len(methods) - 1) / 2) * width
        ax.bar(positions, heights, width * 0.92, color=COLORS[method], label=LABELS[method])
        for x, y in zip(positions, heights):
            if np.isfinite(y):
                ax.text(x, y + 0.02, f"{y:.2f}", ha="center", fontsize=8, color=MUTED)
    count = metrics[methods[0]].n if methods else 0
    ax.set_xticks(np.arange(len(groups)), ["Top-1", f"Top-{k}", "MRR"])
    ax.set(ylim=(0, 1.12), title=f"Source localization on {split} (n={count})")
    ax.legend(frameon=False, loc="upper left", ncol=3)
    _style(ax)
    return fig


def plot_confusion(report: Any, method: str = "gat", channel_names: Sequence[str] | None = None,
                   ax: Any = None) -> Any:
    """Validation confusion matrix of true vs. predicted source."""
    plt = _pyplot()
    matrix = report.confusion(method)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
    else:
        fig = ax.figure
    if matrix is None:
        ax.text(0.5, 0.5, "no validation episodes", ha="center", va="center", color=MUTED)
        ax.set_axis_off()
        return fig
    image = ax.imshow(matrix, cmap="Blues")
    names = list(channel_names or range(len(matrix)))
    ax.set_xticks(range(len(matrix)), names, rotation=45)
    ax.set_yticks(range(len(matrix)), names)
    for (row, column), value in np.ndenumerate(matrix):
        if value:
            ax.text(column, row, str(value), ha="center", va="center",
                    color="white" if value > matrix.max() / 2 else "#0b0b0b", fontsize=9)
    ax.set(title=f"{LABELS.get(method, method)} validation confusion",
           xlabel="predicted source", ylabel="true source")
    fig.colorbar(image, ax=ax, shrink=0.8)
    return fig


def plot_training_report(report: Any, channel_names: Sequence[str] | None = None) -> Any:
    """Dashboard: learning curves, model comparison and both confusion matrices."""
    plt = _pyplot()
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    plot_training_history(report, [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])])
    plot_model_comparison(report, ax=fig.add_subplot(grid[0, 2]))
    plot_confusion(report, "gnn", channel_names, fig.add_subplot(grid[1, 0]))
    plot_confusion(report, "gat", channel_names, fig.add_subplot(grid[1, 1]))
    ax = fig.add_subplot(grid[1, 2])
    ax.set_axis_off()
    short = {"heuristic": "heuristic", "gnn": "GNN", "gat": "GAT"}
    rows = [[row["split"][:5], short[row["method"]], row["n"], f"{row['top1']:.2f}",
             f"{row['mrr']:.2f}", f"{row['nll']:.2f}"] for row in report.summary_table()]
    if rows:
        table = ax.table(cellText=rows, colLabels=["split", "model", "n", "top-1", "MRR", "NLL"],
                         colWidths=[0.15, 0.22, 0.11, 0.15, 0.15, 0.15], loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
    ax.set_title(f"Metrics (split by {report.split_unit})")
    fig.suptitle("GraphProcessModel training quality")
    return fig


def plot_attention(prediction: Any, channel_names: Sequence[str] | None = None,
                   ax: Any = None) -> Any:
    """Heatmap of the episode-specific GAT attention matrix (row i attends to j)."""
    plt = _pyplot()
    attention = np.asarray(prediction.attention)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    else:
        fig = ax.figure
    image = ax.imshow(attention, cmap="Blues", vmin=0, vmax=1)
    names = list(channel_names or range(len(attention)))
    ax.set_xticks(range(len(names)), names, rotation=45)
    ax.set_yticks(range(len(names)), names)
    ax.set(title=f"GAT attention (GAT source: {names[prediction.gat_source]})",
           xlabel="attended neighbor j", ylabel="node i")
    fig.colorbar(image, ax=ax, shrink=0.8)
    return fig


def plot_evaluation(report: Any) -> Any:
    """Detection outcomes per route and n-D localization ranks per method."""
    plt = _pyplot()
    summary = report.summary()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    outcomes = ["hit", "miss", "early_alarm", "false_alarm", "correct_rejection"]
    routes = list(summary)
    width = 0.8 / max(len(routes), 1)
    for index, route in enumerate(routes):
        positions = np.arange(len(outcomes)) + (index - (len(routes) - 1) / 2) * width
        axes[0].bar(positions, [summary[route][name] for name in outcomes], width * 0.92,
                    color=ROUTE_COLORS.get(route, NEUTRAL), label=f"{route} cases")
    axes[0].set_xticks(np.arange(len(outcomes)), [name.replace("_", "\n") for name in outcomes])
    axes[0].set(title="Detection outcomes by route", ylabel="cases")
    axes[0].legend(frameon=False)
    _style(axes[0])
    ax = axes[1]
    nd = summary.get("n-d")
    methods = [m for m in ("heuristic", "gnn", "gat") if nd and nd.get(f"{m}_n")]
    if not methods:
        ax.text(0.5, 0.5, "no localized n-D cases", ha="center", va="center", color=MUTED)
        ax.set_axis_off()
        return fig
    names = ["top1", f"top{report.k}", "mrr"]
    width = 0.8 / len(methods)
    for index, method in enumerate(methods):
        positions = np.arange(len(names)) + (index - (len(methods) - 1) / 2) * width
        heights = [nd[f"{method}_{name}"] for name in names]
        ax.bar(positions, heights, width * 0.92, color=COLORS[method],
               label=f"{LABELS[method]} (n={nd[f'{method}_n']})")
    ax.set_xticks(np.arange(len(names)), ["Top-1", f"Top-{report.k}", "MRR"])
    ax.set(ylim=(0, 1.1), title="n-D source localization")
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    return fig
