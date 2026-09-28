# Using Graph-EVT-agent in your own code: a walkthrough

This is a story-style guide for people who want to **call Graph-EVT-agent from
their own programs**: scripts, notebooks, services or batch jobs. It follows one
project from raw recordings to a trained and evaluated GNN/GAT source model.
Every step has a matching, executed notebook in [`notebooks/`](../notebooks):

| Step | Question | Notebook | Main API |
| --- | --- | --- | --- |
| 1 | Is there an extreme episode, and which channel led it? | [`01_quickstart_1d_nd`](../notebooks/01_quickstart_1d_nd.ipynb) | `GraphEVTPipeline` |
| 2 | How do I get labelled episodes for training? | [`02_label_episodes`](../notebooks/02_label_episodes.ipynb) | `EpisodeLabeler` |
| 3 | Can a GNN/GAT learn a better source model, and how well did training go? | [`03_train_gnn_gat`](../notebooks/03_train_gnn_gat.ipynb) | `ProcessModelTrainer`, `visualization` |
| 4 | How good is the whole system on held-out 1-D and n-D inputs? | [`04_evaluation_orchestration`](../notebooks/04_evaluation_orchestration.ipynb) | `EvaluationOrchestrator` |

The notebooks use the package's synthetic generators
(`graph_evt_agent.synthetic`), so they run offline in about a minute and the
ground truth is known. Swap in your own recordings once the flow makes sense.

---

## 0. Install it into your project

```bash
# from GitHub, with plotting, progress bars and notebook tools
python -m pip install "graph-evt-agent[notebooks] @ git+https://github.com/D2718281828nis/Graph-EVT-agent"

# or a local clone for development
git clone https://github.com/D2718281828nis/Graph-EVT-agent.git
cd Graph-EVT-agent && python -m pip install -e '.[dev,viz,progress]' && pytest
```

| Extra | Adds | Needed for |
| --- | --- | --- |
| core | numpy, scipy, mistralai | detection, graphs, labelling, training, evaluation |
| `[viz]` | matplotlib | `graph_evt_agent.visualization` |
| `[progress]` | tqdm | nicer progress bars (a text bar is used without it) |
| `[edf]` | pyEDFlib | EDF/EDF+/BDF files |
| `[notebooks]` | matplotlib, tqdm, pandas, jupyterlab | the tutorial notebooks |

To open the tutorials: `cd notebooks && jupyter lab`.

---

## 1. First contact: push a 1-D or n-D series

You have a recording. You do not need to tell the library whether it is 1-D or
n-D; the deterministic planner looks at the input and picks the route.

```python
from graph_evt_agent import EVTConfig, GraphConfig, GraphEVTPipeline, load_timeseries

pipeline = GraphEVTPipeline(
    EVTConfig(baseline_size=5_000, tail_quantile=0.95, alarm_probability=0.999,
              top_k=3, persistence=5),
    GraphConfig(method="ar", edge_threshold=0.3),   # or dfa | kuramoto | ensemble | auto
    verbose=True,                                    # progress bar per pipeline stage
)
result = pipeline.run_detailed(load_timeseries("episode.csv"))

result.plan.route              # "1-d" or "n-d"
result.plan.completed_actions  # e.g. inspect_input → … → rank_source
result.plan.stop_reason        # None | "no_evt_detected" | "single_channel_cannot_localize"
result.detection.time_index    # first persistent alarm
result.ranking                 # None for 1-D or no event
```

The only contract you must respect: **the first `baseline_size` samples are
pre-event background.** Scaling, the POT/GPD threshold and the channel graph
are frozen on them.

```python
from graph_evt_agent import visualization as viz
viz.plot_detection(result).savefig("detection.png", dpi=150)
```

---

## 2. Building a training set: label episodes with EVT + graph

A learned source model needs many independent episodes with a known source.
Long recordings usually contain several events, and experts rarely annotate
all of them. `EpisodeLabeler` does the mechanical part:

```python
import pandas as pd
from graph_evt_agent import EpisodeLabeler, LabelingConfig

labeler = EpisodeLabeler(pipeline, LabelingConfig(
    min_confidence=0.5,   # top heuristic weight needed for a pseudo-label
    min_margin=0.2,       # top-1 minus top-2 weight
    refractory=50,        # samples before the detector may re-arm
    match_tolerance=25,   # how far an annotation may be from a detected onset
), verbose=True)

labels = labeler.label(
    {"patient01_run1": load_timeseries("p01_r1.csv"),
     "patient02_run1": load_timeseries("p02_r1.csv")},
    annotations={"patient01_run1": [(12_340, "A3"), (20_115, "A1")]},  # (onset, source)
)
labels.summary()   # {'episodes': …, 'manual': …, 'pseudo': …, 'abstained': …, 'skipped_series': …}
pd.DataFrame(labels.to_records()).to_csv("labels.csv", index=False)
values = load_timeseries("p01_r1.csv").values
viz.plot_labelled_recording(values, labels, "patient01_run1", baseline_size=5_000)
```

For each recording it freezes the EVT threshold and the graph on the baseline,
finds **every** declustered alarm, computes node features
`[peak, energy, latency, degree, neighbor_peak]`, and marks each episode as
`manual`, `pseudo` or `abstained`. One-channel recordings are skipped with a
reason, because a source cannot be localized from one channel.

**What the notebook shows:** on synthetic data where some neighbors respond
louder than the source, confident pseudo-labels are right only about 40% of
the time. Pseudo-labels copy the heuristic ranker; they are a bootstrap, not
ground truth.

---

## 3. Training GNN and GAT, and checking the training went well

```python
from graph_evt_agent import GraphLearningConfig, ProcessModelTrainer

trainer = ProcessModelTrainer(
    GraphLearningConfig(hidden_dim=8, epochs=300, learning_rate=0.02, random_state=42),
    validation_fraction=0.25,   # whole recordings are held out, never single events
    verbose=True,               # progress bar with running loss and validation accuracy
)
report = trainer.train(labels)             # or train(list_of_GraphEpisode, groups=…)
report.summary_table()                     # heuristic vs GNN vs GAT on train and validation
report.best_epoch                          # lowest validation loss → early-stopping hint
viz.plot_training_report(report).savefig("training.png", dpi=150)
```

`GraphProcessModel` holds two small NumPy models:

* **GNN** – node features plus normalized 1-hop and 2-hop neighborhood
  aggregates → nonlinear hidden layer → one logit per node;
* **GAT** – attention over baseline-graph edges only, giving a per-episode
  directed attention matrix → nonlinear hidden layer → one logit per node.

The training dashboard answers four questions:

1. **Is it learning?** Training loss falls, training accuracy rises.
2. **Is it overfitting?** Validation loss (dashed) turns upward while training
   loss keeps falling. The dotted line is `best_epoch`.
3. **Is it worth it?** Validation Top-1/Top-3/MRR against the transparent
   heuristic ranker. If the learned models do not beat it, keep the heuristic.
4. **Where does it fail?** Confusion matrices; errors usually fall on graph
   neighbors of the true source.

In the notebook, 300 epochs overfit (validation loss rises after about 25–60
epochs). Retraining with `epochs=report.best_epoch` lifts validation Top-1 from
0.40 (heuristic) to about 0.55 on this synthetic task. Treat that as a
demonstration of the workflow, not as an expected gain on your data.

Use and ship the model:

```python
from graph_evt_agent import GraphProcessModel

trained = GraphEVTPipeline(evt_config, graph_config, process_model=report.model)
result = trained.run_detailed(new_recording)
result.process.gnn_source, result.process.gat_source, result.process.attention
viz.plot_attention(result.process, result.input_profile.channel_names)

report.model.save("graph_process_model.npz")              # plain .npz, no pickle
model = GraphProcessModel.load("graph_process_model.npz")
```

The feature schema and channel layout must match the training data.

---

## 4. Evaluating on held-out data: 1-D and n-D in one batch

Real test sets mix single-sensor and multichannel inputs, recordings with and
without events, and sometimes inputs with no ground truth at all.
`EvaluationOrchestrator` routes each case by what the planner sees:

```python
from graph_evt_agent import EvaluationCase, EvaluationOrchestrator

cases = [
    EvaluationCase(load_timeseries("test_nd_01.csv"), event_time=5_210, source="A3"),
    EvaluationCase(single_sensor_array, event_time=7_020),    # 1-D: detection only
    EvaluationCase(quiet_recording, event_time=None),         # no event expected
    load_timeseries("unlabelled.csv"),                        # run and describe only
]
report = EvaluationOrchestrator(trained, early_tolerance=0, k=3, verbose=True).evaluate(cases)

report.summary()["1-d"]   # detection_rate, false_alarm_rate, mean/median delay, outcome counts
report.summary()["n-d"]   # + heuristic/gnn/gat top1, top3, mrr
pd.DataFrame(report.to_records())
viz.plot_evaluation(report)
```

Outcomes per case are `hit`, `miss`, `early_alarm`, `false_alarm`,
`correct_rejection` or `unlabelled`. Localization is scored only for n-D hits
with a known source, so a missed detection is never counted as a correct
localization.

---

## 5. Embedding in other programs: progress without a terminal

Every long-running entry point takes `verbose` and `progress_callback`:

| Object | Progress unit |
| --- | --- |
| `GraphEVTPipeline(..., verbose=, progress_callback=)` / `run_detailed(x, verbose=)` | planner actions |
| `EpisodeLabeler(..., verbose=, progress_callback=)` | recordings |
| `ProcessModelTrainer(..., verbose=, progress_callback=)` / `GraphProcessModel.fit(..., verbose=)` | epochs |
| `EvaluationOrchestrator(..., verbose=, progress_callback=)` | cases |

* `verbose=False` (default): silent, suitable for libraries and servers.
* `verbose=True`: one progress bar on stderr (tqdm when installed).
* `verbose=2`: also nested bars, e.g. pipeline stages inside an evaluation.
* `progress_callback(event)`: receives a `ProgressEvent(desc, step, total,
  message, elapsed)` with `event.fraction`. It fires whether or not a bar is
  shown, and includes nested tasks.

```python
import logging
log = logging.getLogger("my_service")

def report_progress(event):
    log.info("%s %d/%d %.0f%% %s", event.desc, event.step, event.total,
             100 * event.fraction, event.message)
    # or: websocket.send(json.dumps(event.__dict__)), job.update_state(meta=…), …

pipeline = GraphEVTPipeline(evt_config, graph_config, progress_callback=report_progress)
```

---

## 6. Optional: plain-language review by LLM agents

After the numbers are computed, `EVTAgentTeam` can have Mistral (or any client
with `complete(system, user) -> str`) explain and check them. The agents never
change thresholds, edges or rankings. See the [README](../README.md) for key
handling and [PIPELINE.md](PIPELINE.md) for the routing.

---

## Checklist before trusting results on your data

- Baseline really precedes the event; configuration frozen before testing.
- Split by subject/device/recording, never by samples of one episode.
- Pseudo-labels used only for bootstrapping; validation and test use manual labels.
- Learned models compared with the heuristic ranker and simple baselines, with
  confidence intervals over independent episodes.
- Attention and graph edges read as associations, not causes.
