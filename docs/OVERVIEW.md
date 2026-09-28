# Graph EVT Agent: capabilities and user guide

This guide summarizes what the `graph-evt-agent` package (v0.1.0) can do and
how to use each part. It complements the Russian [README](../README.md) and the
detailed design notes in [PIPELINE.md](PIPELINE.md).

## In one paragraph

Give the package a multichannel time series (CSV, EDF/BDF, NumPy array,
DataFrame, or dict) whose first `baseline_size` samples are "quiet" background.
It will (1) detect **when** a persistent extreme episode starts, using
extreme-value theory (POT/GPD); (2) build a **graph hypothesis** of how channels
are associated, from the background only, using AR innovations, DFA/cross-DFA,
or wavelet-phase Kuramoto synchrony; (3) **rank which channel** most likely
responded first and strongest; (4) optionally apply a **trained GNN/GAT** to
produce learned source probabilities and an episode-specific attention graph;
and (5) optionally have a team of **LLM reviewer agents** (Mistral, or any
provider you plug in) explain and audit the numeric result in plain language.

All numbers come from deterministic NumPy/SciPy code. The LLM layer only
explains results; it never changes thresholds, edges, or rankings.

```
input ──► InputInspector ──► PipelinePlanner ──► EVT (POT/GPD)
                                                   │ detected & n-D?
                                                   ▼
                          baseline graph: AR | DFA | Kuramoto-wavelet | ensemble
                                                   ▼
                          transparent source ranking ──► [optional] GNN + GAT
                                                   ▼
                          [optional] EVTAgentTeam: input → EVT → graph →
                                     localization reviewers → coordinator
```

## Capability map

| Component | Module | What it does | Key output |
| --- | --- | --- | --- |
| Loading | `io.py` | Wide CSV (with timestamp column auto-detect, ISO-8601 or numeric time) and EDF/EDF+/BDF via `pyEDFlib` | `TimeSeriesInput` |
| Input inspection | `input.py` | Normalizes to `[time, channel]`, validates names, timestamps, regular sampling, missing values (`error` / `drop_rows` / `interpolate`) | `InputProfile` |
| Orchestration | `planner.py` | Deterministic plan → act → observe state machine: chooses 1-D vs n-D route, checks baseline stationarity, records completed steps and why it stopped | `PipelinePlan` |
| EVT detection | `evt.py` | Robust median/MAD standardization on the baseline, Top-k aggregate indicator, GPD tail fit, first run of `persistence` exceedances | `EVTDetection` |
| AR graph | `graph.py` | Correlation of per-channel AR(p) innovations | `GraphResult` |
| DFA graph | `graph.py` | Multiscale DFA exponents + detrended cross-correlation (DCCA-like) | `GraphResult` + `alpha`, `fluctuation` |
| Wavelet–Kuramoto graph | `graph.py` | Morlet wavelet phases, pairwise phase-locking value, circular-shift surrogate test, Kuramoto order parameter | `GraphResult` + surrogate thresholds, order parameter |
| Ensemble / auto | `graph.py` | Intersection of DFA and Kuramoto edges; `auto` picks AR for stationary baselines, ensemble otherwise | `GraphResult` |
| Source ranking | `localization.py` | Scores peak, energy, latency and neighbor agreement; softmax weights | `SourceRanking` |
| GNN / GAT | `learning.py` | Pure-NumPy supervised node classifiers trained on labelled episodes | `GraphProcessPrediction` |
| Agent team | `agents.py` | Role-based LLM reviewers + coordinator over bounded JSON evidence | `AgentReport` |

### EVT detection ("when?")

1. Channel-wise robust z-scores: `|x − median| / (1.4826·MAD)` using the
   baseline only.
2. Per time step, the mean of the `top_k` largest z-scores becomes a scalar
   indicator (so one noisy channel cannot trigger an alarm alone when
   `top_k > 1`).
3. Peaks over the baseline `tail_quantile` are fitted with a generalized Pareto
   distribution; the alarm threshold is the level with exceedance probability
   `1 − alarm_probability`. If too few excesses exist, an empirical quantile is
   used.
4. The event time is the first post-baseline index that starts `persistence`
   consecutive exceedances.

Everything is fit on the baseline, so test/event data cannot leak into the
threshold.

### Graph construction ("where could it spread?")

The graph always uses **only the baseline** and is undirected with self-loops.
Edges are association hypotheses, not causal links.

- **AR (`method="ar"`)** – fits a separate AR(`ar_order`) model to each channel
  and connects channels whose residuals correlate above `edge_threshold`.
  Suitable for roughly stationary signals.
- **DFA (`method="dfa"`)** – integrates each channel, removes local linear
  trends in windows of each `dfa_scales` size, estimates the Hurst-like
  exponent α per channel, and combines (50/50) the mean detrended
  cross-correlation with the similarity of α. Robust to slow trends and
  non-stationarity. Diagnostics: `graph.diagnostics["alpha"]`,
  `["fluctuation"]`, `["scales"]`.
- **Wavelet + Kuramoto (`method="kuramoto"`)** – convolves each channel with a
  complex Morlet wavelet at every period in `wavelet_periods`, extracts phases,
  and computes pairwise phase-locking values (PLV). An edge requires PLV ≥
  `edge_threshold` **and** PLV above the `surrogate_quantile` of
  `surrogate_count` circular-shift surrogates (reproducible via
  `random_state`). The mean Kuramoto order parameter per band is in
  `graph.diagnostics["order_parameter_mean"]`. Good for oscillatory,
  synchronizing systems (EEG, power grids, coupled oscillators).
- **Ensemble (`method="ensemble"`)** – keeps only edges accepted by both DFA and
  Kuramoto (conservative); dependence is their geometric mean.
- **Auto (`method="auto"`, default)** – AR if a four-block mean/variance check
  calls the baseline stationary, otherwise ensemble. For experiments, fix the
  method on training data instead.

You can call the builders directly without detection:

```python
from graph_evt_agent import GraphConfig
from graph_evt_agent.graph import build_graph, dfa_graph, kuramoto_wavelet_graph

g = build_graph(baseline_array, GraphConfig(method="kuramoto", wavelet_periods=(8.0, 16.0)))
print(g.method, g.adjacency.astype(int), g.diagnostics["order_parameter_mean"])
```

### Source ranking ("which node first?")

For a window of 12 samples after the alarm, each channel gets features
`[peak, energy, latency, degree, neighbor_peak]`. The score
`peak + √energy + 0.15·neighbor_peak − 0.5·latency` is turned into softmax
weights. This is a transparent, untrained baseline; weights are relative
scores, not calibrated probabilities. Univariate input is never localized.

### GNN and GAT (learned localization)

`GraphProcessModel` trains two compact models in pure NumPy (Adam, L2,
node-level cross-entropy) from `GraphEpisode(features, adjacency, source)`
objects:

- **GNN** – each node sees its own features plus normalized 1-hop and 2-hop
  neighborhood aggregates, followed by a tanh hidden layer and a per-node logit.
- **GAT** – learns source/target attention vectors with LeakyReLU, masked to
  baseline edges, and classifies from self + attention-weighted neighbor
  features. The returned `attention` matrix `[i, j]` is a directed,
  episode-specific "process graph" (how much node *i* relies on neighbor *j*).

It needs **independent episodes with known source labels**; the package never
invents pseudo-labels. When a fitted model is passed to the pipeline via
`process_model=`, results appear in `run.process`.

### Orchestration and agents

There are two orchestration layers:

1. **Numerical orchestrator (`PipelinePlanner`)** – deterministic. It plans the
   eligible actions (`inspect_input`, `check_stationarity`, `detect_evt`,
   `build_baseline_graph`, `rank_source`, `infer_gnn_gat_process`), then
   records what actually ran and a `stop_reason`
   (`no_evt_detected`, `single_channel_cannot_localize`, or `None`). Inspect it
   at `run.plan`.
2. **LLM review team (`EVTAgentTeam`)** – runs the pipeline, builds a bounded
   JSON evidence summary (no raw samples), and calls reviewer roles in a fixed
   order: input reviewer → EVT reviewer → graph reviewer → localization
   reviewer → coordinator. Graph and localization reviewers are skipped for
   univariate or undetected cases, so a run costs 5 or 3 LLM calls. Every role
   is instructed not to change numbers and to separate statistical rarity from
   causality.

`MistralClient` wraps the official `mistralai` SDK, reads `MISTRAL_API_KEY` from
the environment, retries 429/5xx/connection errors, and raises a sanitized
`MistralAPIError` for 401/403. Any object with
`complete(system: str, user: str) -> str` can replace it (another LLM provider,
a local model, or a fake for tests).

## How to use it

### Install

```bash
git clone https://github.com/D2718281828nis/Graph-EVT-agent.git
cd Graph-EVT-agent
python -m venv .venv && source .venv/bin/activate
python -m pip install -e .          # core: numpy, scipy, mistralai
python -m pip install -e '.[edf]'   # + EDF/BDF reading
python -m pip install -e '.[dev]'   # + pytest, ruff
pytest                              # 26 offline tests, no API key needed
```

Requires Python ≥ 3.10.

### Prepare data

- Shape `[time, channel]`, synchronous samples, at least one signal channel.
- The first `baseline_size` samples (≥ 20, and enough for the largest DFA
  scale ×2) must be pre-event background, followed by at least one more sample.
- CSV must be **wide** (`timestamp,A1,A2,...`), with a unique header, numeric
  values, strictly increasing time. Long format must be pivoted first.
- EDF channels must share sampling frequency and length (no hidden resampling).
- Missing values raise an error unless you choose
  `InputConfig(missing="drop_rows" | "interpolate")`.

### 1. Numerical pipeline (no LLM, no network)

```python
from graph_evt_agent import EVTConfig, GraphConfig, GraphEVTPipeline, load_timeseries

pipeline = GraphEVTPipeline(
    EVTConfig(baseline_size=5_000, tail_quantile=0.95,
              alarm_probability=0.999, top_k=3, persistence=5),
    GraphConfig(method="ensemble", edge_threshold=0.4,
                dfa_scales=(8, 16, 32, 64), wavelet_periods=(8.0, 16.0, 32.0),
                surrogate_count=199, random_state=42),
)
run = pipeline.run_detailed("episode.csv")   # or a .edf/.bdf path, array, DataFrame

print(run.input_profile.channel_names)
print(run.plan.completed_actions, run.plan.stop_reason)
print("event at", run.detection.time_index, "threshold", run.detection.alarm_threshold)
if run.ranking is not None:
    names = run.input_profile.channel_names
    for node, w in zip(run.ranking.node_ids, run.ranking.probabilities):
        print(names[node], round(float(w), 3))
```

`pipeline.run(...)` returns the older `(detection, graph, ranking)` tuple.

In-memory data works too:

```python
from graph_evt_agent import TimeSeriesInput
run = pipeline.run_detailed(TimeSeriesInput(values=X, timestamps=t, channel_names=("A1", "A2", "A3")))
run = pipeline.run_detailed({"A1": a1, "A2": a2})      # dict of channels
run = pipeline.run_detailed(df)                          # pandas DataFrame
```

### 2. Train and use the GNN/GAT

```python
from graph_evt_agent import GraphEpisode, GraphLearningConfig, GraphProcessModel

episodes = []
for series, known_source in labelled_training_episodes:     # your labelled data
    r = pipeline.run_detailed(series)
    if r.ranking is not None:
        episodes.append(GraphEpisode(r.ranking.features, r.graph.adjacency, known_source))

model = GraphProcessModel(GraphLearningConfig(hidden_dim=8, epochs=300, random_state=42)).fit(episodes)

learned = GraphEVTPipeline(pipeline.evt_config, pipeline.graph_config, process_model=model)
res = learned.run_detailed(new_series)
print(res.ranking.source, res.process.gnn_source, res.process.gat_source)
print(res.process.attention)   # episode-specific directed attention graph
```

All episodes must have the same channel count/order and feature schema.
Evaluate on held-out subjects/devices against the transparent ranker before
trusting the learned model.

### 3. Add the LLM reviewer team

```bash
read -s MISTRAL_API_KEY && export MISTRAL_API_KEY   # never hard-code the key
```

```python
from graph_evt_agent.agents import EVTAgentTeam, MistralClient, MistralAPIError

team = EVTAgentTeam(pipeline=pipeline, client=MistralClient(model="mistral-small-latest"))
try:
    report = team.run("episode.csv", task="Find the probable first contact of the episode")
    print(report.final_report)
    print(report.graph_review)
    numbers = report.result           # full JSON-safe numeric result
except MistralAPIError as err:
    print(err)                        # keep pipeline.run_detailed() output as fallback
```

Prompts are currently written in Russian, so reviews usually come back in
Russian; the `task` text can be in any language.

Use another provider by implementing one method:

```python
class MyClient:
    def complete(self, system: str, user: str) -> str:
        return my_llm.chat(system=system, user=user)

team = EVTAgentTeam(pipeline, MyClient())
```

In GitHub Actions, store the key as the `MISTRAL_API_KEY` repository secret and
pass it only to the step that needs it.

## Typical use cases

- **EEG/iEEG/MEG**: detect a seizure-like or interictal extreme and rank
  candidate onset electrodes; Kuramoto/wavelet graphs capture band-limited
  synchrony.
- **Industrial sensor networks / SCADA**: find when a multi-sensor fault began
  and which sensor deviated first; DFA handles drifting baselines.
- **Power grids and coupled oscillators**: phase-locking graph and order
  parameter describe synchronization before a disturbance.
- **Research benchmarking**: a leakage-safe, reproducible baseline to compare
  against learned localizers (GNN/GAT, MLP, GCN, shuffled-graph controls).

## Limits to keep in mind

- Rare ≠ pathological; a graph edge ≠ causation; first-to-deviate ≠ physical
  source.
- POT assumes roughly independent exceedances; check tail fit and consider
  declustering for strongly autocorrelated data.
- Choose `top_k`, thresholds, graph method, DFA scales and wavelet periods on
  training data, then freeze them.
- Ranking weights are uncalibrated; the GNN/GAT needs many independent labelled
  episodes and gains must be demonstrated on held-out data.
- Directed graphs, automatic AR order selection, resampling and long-format
  CSV are not supported.
- LLM reviews are unverified natural language; if any LLM call fails,
  `EVTAgentTeam.run` raises (no partial report).
- Not a medical or industrial decision system on its own.
