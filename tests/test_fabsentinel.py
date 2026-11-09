import numpy as np
import pytest

from fabsentinel.cli import main
from fabsentinel.diagnose import SignatureDiagnoser
from fabsentinel.features import feature_table, wafer_features
from fabsentinel.model import MSPCDetector
from fabsentinel.simulator import FAULTS, EtchChamberSimulator


@pytest.fixture(scope="module")
def trained():
    sim = EtchChamberSimulator(seed=11)
    det = MSPCDetector().fit(feature_table(sim.run_lot(200)))
    return sim, det


def test_simulator_is_deterministic():
    a = EtchChamberSimulator(seed=3).run_wafer("chamber_leak", 1.0).trace
    b = EtchChamberSimulator(seed=3).run_wafer("chamber_leak", 1.0).trace
    assert a.equals(b)


def test_unknown_fault_rejected():
    with pytest.raises(ValueError):
        EtchChamberSimulator().run_wafer("gremlins")


def test_cf4_drift_stretches_endpoint():
    sim = EtchChamberSimulator(seed=5)
    normal = np.mean([sim.run_wafer().meta["endpoint_s"] for _ in range(20)])
    drift = np.mean([sim.run_wafer("cf4_mfc_drift", 1.5).meta["endpoint_s"] for _ in range(20)])
    assert drift > normal + 1


def test_features_cover_every_step_and_endpoint():
    f = wafer_features(EtchChamberSimulator(seed=1).run_wafer().trace)
    assert {k.split(".")[0] for k in f} == {"stab", "main", "over"}
    assert 55 <= f["main.endpoint_s"] <= 65


def test_false_alarm_rate_is_low(trained):
    sim, det = trained
    scores = det.score(feature_table(sim.run_lot(300)))
    assert np.mean([s.flagged for s in scores]) < 0.03


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_strong_faults_are_caught_and_diagnosed(trained, fault):
    sim, det = trained
    scores = det.score(feature_table([sim.run_wafer(fault, 1.3) for _ in range(10)]))
    assert all(s.flagged for s in scores)
    diag = SignatureDiagnoser()
    correct = sum(diag.diagnose(s.z).cause == fault for s in scores)
    assert correct >= 8


def test_diagnosis_confidences_sum_to_one(trained):
    sim, det = trained
    s = det.score(feature_table([sim.run_wafer("chiller_fault", 1.2)]))[0]
    dx = SignatureDiagnoser().diagnose(s.z)
    assert abs(sum(c for _, c in dx.ranked) - 1) < 1e-6
    assert dx.action


def test_cli_diagnose(capsys):
    assert main(["diagnose", "throttle_valve_sticking", "--severity", "1.2"]) == 0
    out = capsys.readouterr().out
    assert "throttle_valve_sticking" in out and "First action" in out


def test_cli_benchmark_writes_report(tmp_path):
    assert main(["benchmark", "--wafers", "150", "--out", str(tmp_path)]) == 0
    html = (tmp_path / "report.html").read_text()
    assert "<svg" in html and "Top alarms" in html
    assert (tmp_path / "metrics.json").exists()
