"""Build the tutorial notebooks in notebooks/.

Usage: python scripts/build_notebooks.py [--execute]
With --execute (needs nbclient and ipykernel) outputs are stored, so the
notebooks render on GitHub.
"""
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1] / "notebooks"
ROOT.mkdir(exist_ok=True)
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

SETUP = """\
# If the package is not installed yet (run once, then restart the kernel):
# %pip install "graph-evt-agent[notebooks] @ git+https://github.com/D2718281828nis/Graph-EVT-agent"
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42"""

notebooks = {}

notebooks["01_quickstart_1d_nd.ipynb"] = [
    md("""# 1. Quick start: one call for 1-D and n-D series

This notebook shows the smallest useful integration of **Graph-EVT-agent** into your
own code. You push a time series into `GraphEVTPipeline.run_detailed`; the deterministic
planner decides the route:

* **1-D** (one channel) → input check → EVT detection → stop (a single channel cannot be localized);
* **n-D** (several channels) → input check → EVT detection → baseline graph → source ranking.

We use the package's synthetic generators so every number is reproducible and the true
onset and source are known. Replace them with `load_timeseries("your.csv")` for real data."""),
    code(SETUP),
    code("""from graph_evt_agent import EVTConfig, GraphConfig, GraphEVTPipeline, TimeSeriesInput
from graph_evt_agent.synthetic import make_propagation_episode, make_univariate_episode
from graph_evt_agent import visualization as viz

multichannel = make_propagation_episode(n_channels=6, source=2, baseline_size=400, seed=SEED)
univariate = make_univariate_episode(baseline_size=400, seed=SEED)
print("n-D:", multichannel.values.shape, "true onset", multichannel.event_time,
      "true source", multichannel.channel_names[multichannel.source])
print("1-D:", univariate.values.shape, "true onset", univariate.event_time)"""),
    md("""## Configure once, reuse everywhere

`baseline_size` samples at the start **must** be pre-event background: robust scaling,
the POT/GPD threshold and the channel graph are all frozen on it. `verbose=True` draws a
progress bar (tqdm if installed, otherwise a plain text bar) so you can see which stage runs."""),
    code("""pipeline = GraphEVTPipeline(
    EVTConfig(baseline_size=400, tail_quantile=0.95, alarm_probability=0.999,
              top_k=3, persistence=3),
    GraphConfig(method="ar", edge_threshold=0.2, random_state=SEED),
    verbose=True,
)
nd = pipeline.run_detailed(multichannel.as_input())   # values + channel names
od = pipeline.run_detailed(univariate.values[:, 0])"""),
    md("## Inspect the route the planner took"),
    code("""pd.DataFrame([
    {"input": "n-D", "route": nd.plan.route, "detected": nd.detection.detected,
     "alarm index": nd.detection.time_index, "stop reason": nd.plan.stop_reason,
     "completed actions": " → ".join(nd.plan.completed_actions),
     "top source": nd.input_profile.channel_names[nd.ranking.source]},
    {"input": "1-D", "route": od.plan.route, "detected": od.detection.detected,
     "alarm index": od.detection.time_index, "stop reason": od.plan.stop_reason,
     "completed actions": " → ".join(od.plan.completed_actions), "top source": None},
])"""),
    md("""## Visualize detection and ranking

The upper panel shows the robust top-k indicator against the POT/GPD threshold fitted on the
baseline; the lower panel (n-D only) shows the transparent source ranking. Weights are relative
scores, not calibrated probabilities of physical causation."""),
    code("fig = viz.plot_detection(nd)"),
    code("fig = viz.plot_detection(od)"),
    md("""## Your own data

```python
from graph_evt_agent import load_timeseries
source = load_timeseries("episode.csv")            # wide CSV: timestamp,A1,A2,...
source = load_timeseries("record.edf")             # needs the [edf] extra
result = pipeline.run_detailed(source, verbose=False)
```

A pandas `DataFrame`, a `dict[str, array]` or a NumPy `[time, channel]` array work as well.
Continue with **02_label_episodes.ipynb** to turn long recordings into labelled episodes."""),
]

notebooks["02_label_episodes.ipynb"] = [
    md("""# 2. Labelling episodes with EVT and the channel graph

A GNN/GAT needs many **independent labelled episodes**. `EpisodeLabeler` builds them from long
recordings:

1. freeze robust scaling, the POT/GPD threshold and the channel graph on each recording's baseline;
2. find *every* persistent alarm (declustered with a refractory period), not only the first;
3. compute node features `[peak, energy, latency, degree, neighbor_peak]` for every event;
4. assign a label origin:
   * `manual` – an expert annotation `(event_time, source)` matches the detected onset;
   * `pseudo` – no annotation, but the transparent ranker is confident (weight and margin thresholds);
   * `abstained` – ambiguous, kept for inspection but not used for training.

> Pseudo-labels copy the transparent ranker. They are useful for bootstrapping, but a model
> trained only on them imitates the ranker. Always evaluate against manual labels."""),
    code(SETUP),
    code("""from graph_evt_agent import (EVTConfig, GraphConfig, GraphEVTPipeline,
                             EpisodeLabeler, LabelingConfig)
from graph_evt_agent.synthetic import make_recording
from graph_evt_agent import visualization as viz

# 12 recordings with 10 events each. gain_jitter makes some neighbors respond louder
# than the true source, so "largest peak" is not always right – like real sensors.
recordings = {f"rec{i:02d}": make_recording(n_events=10, n_channels=6, baseline_size=400,
                                            amplitude=8, gain_jitter=0.8, delay_per_hop=3,
                                            seed=SEED + i)
              for i in range(12)}
pipeline = GraphEVTPipeline(EVTConfig(baseline_size=400, persistence=3),
                            GraphConfig(method="ar", edge_threshold=0.2, random_state=SEED))
labeler = EpisodeLabeler(pipeline, LabelingConfig(min_confidence=0.5, min_margin=0.2,
                                                  refractory=50), verbose=True)"""),
    md("## Pseudo-labels only (no annotations)"),
    code("""pseudo = labeler.label({key: rec.as_input() for key, rec in recordings.items()})
print(pseudo.summary())
table = pd.DataFrame(pseudo.to_records())
table.head(8)"""),
    code("""fig = viz.plot_labelled_recording(recordings["rec00"].values, pseudo, "rec00",
                                  channel_names=recordings["rec00"].channel_names,
                                  baseline_size=400)"""),
    md("""## How good are the pseudo-labels?

Because the data are synthetic we know the truth, so we can measure what you usually cannot:
how often the confident heuristic label is correct."""),
    code("""def true_source(series_id, event_time):
    rec = recordings[series_id]
    nearest = int(np.argmin([abs(onset - event_time) for onset in rec.event_times]))
    return rec.sources[nearest]

table["true_source"] = [true_source(s, t) for s, t in zip(table.series_id, table.event_time)]
labelled = table[table.origin == "pseudo"]
print(f"pseudo-label accuracy: {(labelled.source == labelled.true_source).mean():.2f} "
      f"on {len(labelled)} episodes; abstained: {(table.origin == 'abstained').sum()}")"""),
    md("""## Adding expert annotations

Annotations are `(event_time, source)` pairs per recording; `source` may be a channel index or
name. A detected onset within `match_tolerance` samples inherits the manual label. Here we
annotate every event with the ground truth, as an expert would after review."""),
    code("""annotations = {key: list(zip(rec.event_times, rec.sources)) for key, rec in recordings.items()}
manual = labeler.label({key: rec.as_input() for key, rec in recordings.items()}, annotations)
print(manual.summary())
# Save for the next notebook / other programs
pd.DataFrame(manual.to_records()).to_csv("labelled_episodes.csv", index=False)"""),
    md("""In practice you would annotate only part of the recordings, train on
`origins=("manual", "pseudo")`, and keep manually labelled recordings for validation.
Continue with **03_train_gnn_gat.ipynb**."""),
]

notebooks["03_train_gnn_gat.ipynb"] = [
    md("""# 3. Training the GNN and GAT process model and visualizing its quality

`ProcessModelTrainer` fits `GraphProcessModel` on labelled episodes:

* the **GNN** uses each node's features plus 1-hop and 2-hop neighborhood aggregates;
* the **GAT** learns attention restricted to baseline graph edges and returns a per-episode
  directed attention matrix;
* validation holds out **whole recordings**, so events from one recording never appear in both
  train and validation;
* every epoch records loss and top-1 accuracy for train and validation, and final metrics are
  compared with the untrained transparent ranker (Top-1, Top-3, MRR, NLL)."""),
    code(SETUP),
    code("""from graph_evt_agent import (EVTConfig, GraphConfig, GraphEVTPipeline, EpisodeLabeler,
                             GraphLearningConfig, GraphProcessModel, ProcessModelTrainer,
                             TimeSeriesInput)
from graph_evt_agent.synthetic import make_recording
from graph_evt_agent import visualization as viz

recordings = {f"rec{i:02d}": make_recording(n_events=10, n_channels=6, baseline_size=400,
                                            amplitude=8, gain_jitter=0.8, delay_per_hop=3,
                                            seed=SEED + i)
              for i in range(12)}
annotations = {key: list(zip(rec.event_times, rec.sources)) for key, rec in recordings.items()}
pipeline = GraphEVTPipeline(EVTConfig(baseline_size=400, persistence=3),
                            GraphConfig(method="ar", edge_threshold=0.2, random_state=SEED))
labeler = EpisodeLabeler(pipeline)
values = {key: rec.as_input() for key, rec in recordings.items()}
manual = labeler.label(values, annotations)
pseudo = labeler.label(values)
print("manual:", manual.summary())
print("pseudo:", pseudo.summary())"""),
    md("## Train on manual labels (with a progress bar)"),
    code("""trainer = ProcessModelTrainer(
    GraphLearningConfig(hidden_dim=8, epochs=300, learning_rate=0.02, random_state=SEED),
    validation_fraction=0.25, random_state=SEED, verbose=True,
)
report = trainer.train(manual)
pd.DataFrame(report.summary_table()).round(3)"""),
    md("""## Training-quality dashboard

* **Learning curves** – solid = train, dashed = validation. A validation loss that rises while the
  training loss keeps falling means overfitting: reduce `epochs`/`hidden_dim` or add episodes.
* **Model comparison** – learned models are only worth using if they beat the transparent ranker
  on validation.
* **Confusion matrices** – which true sources get confused with which (usually graph neighbors)."""),
    code("fig = viz.plot_training_report(report, channel_names=recordings['rec00'].channel_names)"),
    md("""## Early stopping from the learning curves

The dotted line marks `report.best_epoch`, the epoch with the lowest validation loss. Retraining
with that many epochs is a simple early-stopping rule. Strictly, choosing it on the validation
set makes validation optimistic – confirm the final choice on an untouched test set
(notebook 4)."""),
    code("""print("best epoch:", report.best_epoch)
early = ProcessModelTrainer(
    GraphLearningConfig(hidden_dim=8, epochs=report.best_epoch, random_state=SEED),
    validation_fraction=0.25, random_state=SEED).train(manual)
pd.DataFrame(early.summary_table()).query("split == 'validation'").round(3)"""),
    code("fig = viz.plot_training_history(early)"),
    md("""## Pseudo-labels: the teacher ceiling

Training on pseudo-labels and validating on **manual** labels shows the limitation stated in
notebook 2: the student learns to imitate the heuristic, so it cannot be expected to beat it."""),
    code("""pseudo_train = [e for e in pseudo.select() if e.series_id in {"rec00", "rec01", "rec02", "rec03",
                                                              "rec04", "rec05", "rec06", "rec07"}]
manual_val = [e.to_graph_episode() for e in manual.select()
              if e.series_id in {"rec08", "rec09", "rec10", "rec11"}]
pseudo_report = ProcessModelTrainer(GraphLearningConfig(epochs=300, random_state=SEED)).train(
    [e.to_graph_episode() for e in pseudo_train], validation=manual_val)
pd.DataFrame(pseudo_report.summary_table()).query("split == 'validation'").round(3)"""),
    code("fig = viz.plot_training_history(pseudo_report)"),
    md("## Use, inspect and save the trained model"),
    code("""trained = GraphEVTPipeline(pipeline.evt_config, pipeline.graph_config, process_model=early.model)
episode = recordings["rec11"]
result = trained.run_detailed(TimeSeriesInput(episode.values[:700], channel_names=episode.channel_names))
names = episode.channel_names
print("true source:", names[episode.sources[0]],
      "| heuristic:", names[result.ranking.source],
      "| GNN:", names[result.process.gnn_source],
      "| GAT:", names[result.process.gat_source])
fig = viz.plot_attention(result.process, names)"""),
    code("""early.model.save("graph_process_model.npz")
restored = GraphProcessModel.load("graph_process_model.npz")
np.allclose(restored.predict(result.ranking.features, result.graph.adjacency).gat_probabilities,
            result.process.gat_probabilities)"""),
    md("""Attention is a learned predictive weight, **not** proof of physical direction or causation.
Continue with **04_evaluation_orchestration.ipynb**."""),
]

notebooks["04_evaluation_orchestration.ipynb"] = [
    md("""# 4. Evaluation orchestration for mixed 1-D and n-D inputs

`EvaluationOrchestrator` accepts any mix of inputs. For each case it runs the frozen pipeline,
reads the planner's route and scores only what that route supports:

| Route | Detection metrics | Localization metrics |
| --- | --- | --- |
| 1-D | hit / miss / early alarm / false alarm / correct rejection, delay | – (not identifiable) |
| n-D | same | rank of the true source for heuristic, GNN and GAT → Top-1, Top-3, MRR |

Cases without ground truth are still run and reported as `unlabelled`."""),
    code(SETUP),
    code("""from graph_evt_agent import (EVTConfig, GraphConfig, GraphEVTPipeline, EpisodeLabeler,
                             EvaluationCase, EvaluationOrchestrator, GraphLearningConfig,
                             ProcessModelTrainer)
from graph_evt_agent.synthetic import (make_propagation_episode, make_recording,
                                       make_univariate_episode)
from graph_evt_agent import visualization as viz

evt, graph = EVTConfig(baseline_size=400, persistence=3), GraphConfig(method="ar", edge_threshold=0.2)
# Train a process model on separate recordings (see notebook 3)
train_recordings = {f"rec{i}": make_recording(10, 6, amplitude=8, gain_jitter=0.8, seed=100 + i)
                    for i in range(10)}
labels = EpisodeLabeler(GraphEVTPipeline(evt, graph)).label(
    {k: r.as_input() for k, r in train_recordings.items()},
    {k: list(zip(r.event_times, r.sources)) for k, r in train_recordings.items()})
model = ProcessModelTrainer(GraphLearningConfig(epochs=80, random_state=SEED)).train(labels).model"""),
    md("## Push a mixed batch of test inputs"),
    code("""cases = []
for i in range(30):                                   # n-D with known source
    ep = make_propagation_episode(6, source=i % 6, amplitude=8, gain_jitter=0.8, seed=500 + i)
    cases.append(EvaluationCase.from_synthetic(ep, case_id=f"nd_{i}"))
for i in range(10):                                   # 1-D with an event
    cases.append(EvaluationCase.from_synthetic(make_univariate_episode(seed=600 + i), f"1d_event_{i}"))
for i in range(10):                                   # 1-D without an event
    ep = make_univariate_episode(seed=700 + i, amplitude=0)
    cases.append(EvaluationCase(ep.values[:, 0], event_time=None, case_id=f"1d_quiet_{i}"))
cases.append(make_propagation_episode(6, source=1, seed=900).values)  # raw input, no truth

orchestrator = EvaluationOrchestrator(
    GraphEVTPipeline(evt, graph, process_model=model), early_tolerance=0, k=3, verbose=True)
report = orchestrator.evaluate(cases)"""),
    code("pd.DataFrame(report.to_records()).head(10)"),
    code("pd.DataFrame(report.summary()).round(3)"),
    code("fig = viz.plot_evaluation(report)"),
    md("""## Embedding progress in another program

`verbose=2` also shows nested bars (each case's pipeline stages). For GUIs, services or logs,
pass a `progress_callback`; it receives `ProgressEvent(desc, step, total, message, elapsed)`."""),
    code("""import logging
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger("my_app")

def on_progress(event):
    log.info("%s %d/%d (%.0f%%) %s", event.desc, event.step, event.total,
             100 * event.fraction, event.message)

quiet = EvaluationOrchestrator(GraphEVTPipeline(evt, graph), progress_callback=on_progress)
_ = quiet.evaluate(cases[:3])"""),
    md("""## Good practice

* Freeze every configuration (baseline length, tail quantile, graph method, labelling
  thresholds, model hyper-parameters) **before** the final evaluation.
* Split by subject/device/recording, never by samples of one episode.
* Report confidence intervals over independent episodes and compare against the heuristic
  and simple baselines. Synthetic scores here demonstrate behavior, not real-world accuracy."""),
]

for name, cells in notebooks.items():
    nb = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"}})
    if "--execute" in sys.argv:
        NotebookClient(nb, timeout=600, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, ROOT / name)
    print("wrote", name)
