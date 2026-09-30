import io

import numpy as np
import pytest

from graph_evt_agent import (
    EpisodeLabeler, EvaluationCase, EvaluationOrchestrator, EVTConfig, GraphConfig,
    GraphEVTPipeline, GraphLearningConfig, LabelingConfig, ProcessModelTrainer, Progress,
    TemporalGraphConfig, TimeSeriesInput, build_temporal_graph, ranking_metrics,
)
from graph_evt_agent.evt import detect, find_events
from graph_evt_agent.progress import _TextBar
from graph_evt_agent.synthetic import (
    make_propagation_episode, make_recording, make_univariate_episode,
)
from graph_evt_agent.training import split_episodes


def pipeline(**kwargs) -> GraphEVTPipeline:
    return GraphEVTPipeline(EVTConfig(baseline_size=400, persistence=3),
                            GraphConfig(method="ar", edge_threshold=0.2), **kwargs)


def test_progress_callback_reports_every_pipeline_action():
    events = []
    episode = make_propagation_episode(5, source=1, seed=3)
    result = pipeline(progress_callback=events.append).run_detailed(episode.values)
    assert [event.step for event in events] == list(range(1, len(events) + 1))
    assert len(events) == len(result.plan.initial_actions)
    assert events[-1].fraction == 1.0


def test_progress_stops_early_for_univariate_input_and_text_bar_renders():
    events = []
    univariate = make_univariate_episode(seed=1)
    pipeline(progress_callback=events.append).run_detailed(univariate.values[:, 0])
    assert events[-1].fraction == 1.0 and "single_channel" in events[-1].message
    stream = io.StringIO()
    bar = _TextBar(4, "demo", stream)
    bar.update(2, "half")
    bar.close()
    assert "demo: [" in stream.getvalue() and "2/4 half" in stream.getvalue()
    silent = io.StringIO()
    with Progress(3, "quiet", verbose=False, stream=silent) as progress:
        progress.advance()
    assert silent.getvalue() == ""


def test_find_events_declusters_every_event_in_a_recording():
    recording = make_recording(4, 5, seed=2)
    config = EVTConfig(baseline_size=400, persistence=3)
    onsets = find_events(detect(recording.values, config), config, refractory=50)
    assert len(onsets) == 4
    assert all(abs(found - true) <= 5 for found, true in zip(onsets, recording.event_times))


def test_labeler_uses_annotations_pseudo_labels_and_skips_univariate():
    recording = make_recording(4, 5, seed=2)
    names = tuple(f"ch{node}" for node in range(5))
    source = TimeSeriesInput(recording.values, channel_names=names)
    first = (recording.event_times[0], names[recording.sources[0]])
    labels = EpisodeLabeler(pipeline()).label(
        {"rec": source, "flat": make_univariate_episode(seed=0).values[:, 0]},
        annotations={"rec": [first]},
    )
    assert labels.summary()["episodes"] == 4
    assert labels.episodes[0].origin == "manual"
    assert labels.episodes[0].source == recording.sources[0]
    assert {item.origin for item in labels.episodes[1:]} <= {"pseudo", "abstained"}
    assert labels.skipped == (("flat", "univariate: source is not identifiable"),)
    assert set(labels.groups()) == {"rec"}
    assert labels.to_records()[0]["source_name"] == first[1]
    strict = EpisodeLabeler(pipeline(), LabelingConfig(min_confidence=1.0, min_margin=1.0))
    assert all(item.origin == "abstained" for item in strict.label_series(source).episodes)


def test_trainer_splits_by_recording_and_reports_learning_curves():
    recordings = {f"rec{i}": make_recording(4, 5, seed=i) for i in range(6)}
    annotations = {key: list(zip(r.event_times, r.sources)) for key, r in recordings.items()}
    labels = EpisodeLabeler(pipeline()).label(
        {key: r.values for key, r in recordings.items()}, annotations)
    report = ProcessModelTrainer(GraphLearningConfig(epochs=40, random_state=1),
                                 validation_fraction=0.34).train(labels)
    assert report.split_unit == "group"
    assert report.n_train + report.n_validation == len(labels.training_episodes())
    assert len(report.history["epoch"]) == 40
    assert np.isfinite(report.history["gat_val_loss"]).all()
    assert report.history["gnn_train_loss"][-1] < report.history["gnn_train_loss"][0]
    assert {row["method"] for row in report.summary_table()} == {"heuristic", "gnn", "gat"}
    assert report.confusion("gnn").sum() == report.n_validation


def test_split_and_metrics_are_leak_free_and_correct():
    train, validation, unit = split_episodes(list(range(6)), list("aabbcc"), 0.34, 0)
    groups = list("aabbcc")
    assert unit == "group" and not {groups[i] for i in train} & {groups[i] for i in validation}
    metrics = ranking_metrics([np.array([0.7, 0.2, 0.1]), np.array([0.2, 0.3, 0.5])], [0, 1], k=1)
    assert metrics.top1 == 0.5 and metrics.topk == 0.5 and metrics.mrr == 0.75


def test_evaluation_routes_1d_and_nd_cases():
    cases = [
        EvaluationCase.from_synthetic(make_propagation_episode(5, source=2, seed=4)),
        EvaluationCase.from_synthetic(make_univariate_episode(seed=5)),
        EvaluationCase(make_univariate_episode(seed=6, amplitude=0).values[:, 0]),
        make_propagation_episode(5, source=0, seed=7).values,
    ]
    events = []
    report = EvaluationOrchestrator(pipeline(), progress_callback=events.append).evaluate(cases)
    routes = [case.route for case in report.cases]
    assert routes == ["n-d", "1-d", "1-d", "n-d"]
    assert [case.outcome for case in report.cases] == [
        "hit", "hit", "correct_rejection", "unlabelled"]
    assert report.cases[0].ranks["heuristic"] >= 1
    summary = report.summary()
    assert summary["1-d"]["detection_rate"] == 1.0
    assert summary["n-d"]["heuristic_n"] == 1
    assert len(report.to_records()) == 4
    assert any(event.desc == "Evaluating cases" for event in events)


def test_visualization_returns_figures():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from graph_evt_agent import visualization as viz

    recordings = {f"rec{i}": make_recording(3, 4, seed=i) for i in range(3)}
    labels = EpisodeLabeler(pipeline()).label({k: r.values for k, r in recordings.items()})
    report = ProcessModelTrainer(GraphLearningConfig(epochs=5)).train(labels)
    trained = GraphEVTPipeline(EVTConfig(baseline_size=400, persistence=3),
                               GraphConfig(method="ar", edge_threshold=0.2),
                               process_model=report.model)
    episode = make_propagation_episode(4, source=1, seed=9)
    evaluation = EvaluationOrchestrator(trained).evaluate(
        [EvaluationCase.from_synthetic(episode)])
    result = evaluation.cases[0].result
    figures = [
        viz.plot_training_report(report), viz.plot_training_history(report),
        viz.plot_model_comparison(report), viz.plot_confusion(report),
        viz.plot_labelled_recording(recordings["rec0"].values, labels, "rec0", baseline_size=400),
        viz.plot_detection(result), viz.plot_attention(result.process),
        viz.plot_evaluation(evaluation),
    ]
    series = recordings["rec0"].values[:, 0]
    for temporal in (TemporalGraphConfig(baseline_size=200),
                     TemporalGraphConfig(baseline_size=200, edge_method="wavelet", prune="knn")):
        figures.append(viz.plot_temporal_graph(series, build_temporal_graph(series, temporal),
                                               event_time=int(recordings["rec0"].event_times[0])))
    assert all(hasattr(figure, "savefig") for figure in figures)
    import matplotlib.pyplot as plt
    plt.close("all")
