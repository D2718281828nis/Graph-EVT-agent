import numpy as np
import pytest

from graph_evt_agent import EVTConfig, GraphEVTPipeline, InputInspector, load_csv, load_timeseries


def test_csv_loader_reads_wide_numeric_channels(tmp_path):
    path = tmp_path / "episode.csv"
    path.write_text("timestamp,A1,A2\n0.0,1.0,2.0\n0.5,3.0,4.0\n1.0,5.0,6.0\n")
    source = load_csv(path)
    prepared = InputInspector().inspect(source)
    assert prepared.profile.channel_names == ("A1", "A2")
    assert prepared.profile.sampling_interval == 0.5
    assert prepared.values.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def test_csv_loader_supports_iso_timestamps_and_missing_signal(tmp_path):
    path = tmp_path / "episode.csv"
    path.write_text(
        "datetime,signal\n2026-01-01T00:00:00,1\n"
        "2026-01-01T00:00:01,\n2026-01-01T00:00:02,3\n"
    )
    source = load_timeseries(path)
    assert np.isnan(source.values[1, 0])
    with pytest.raises(ValueError, match="missing"):
        InputInspector().inspect(source)


def test_csv_loader_rejects_long_layout(tmp_path):
    path = tmp_path / "long.csv"
    path.write_text("timestamp,channel,value\n0,A1,1\n0,A2,2\n")
    with pytest.raises(ValueError):
        # String channel identifiers cannot be interpreted as signal samples.
        load_csv(path)


def test_dispatch_rejects_undocumented_file_type(tmp_path):
    with pytest.raises(ValueError, match="supported input file extensions"):
        load_timeseries(tmp_path / "episode.txt")


def test_pipeline_accepts_csv_path_directly(tmp_path):
    rng = np.random.default_rng(8)
    data = rng.normal(size=180)
    data[150:155] += 20
    path = tmp_path / "one_channel.csv"
    rows = ["time,signal"] + [f"{index},{value}" for index, value in enumerate(data)]
    path.write_text("\n".join(rows))
    result = GraphEVTPipeline(EVTConfig(120, persistence=2)).run_detailed(path)
    assert result.input_profile.kind == "univariate"
    assert result.detection.detected
    assert result.graph is None
