import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
SCENE = REPO / "src" / "mjlab_microduck" / "robot" / "microduck" / "scene.xml"


@pytest.fixture(scope="module")
def classic_controller():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            "classic_controller", SCRIPTS / "classic_controller.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


@pytest.fixture(scope="module")
def distill_classic(classic_controller):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            "distill_classic", SCRIPTS / "distill_classic.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


@pytest.mark.parametrize("initial_pitch", [-0.05, 0.05])
def test_pitch_disturbance_is_recovered_without_policy_weights(
    classic_controller, initial_pitch
):
    metrics = classic_controller.run_rollout(duration=5.0, initial_pitch=initial_pitch)

    assert metrics["policy_weights_loaded"] is False
    assert metrics["fell"] is False
    assert metrics["minimum_base_height_m"] > 0.1
    assert abs(metrics["final_pitch_rad"]) < abs(initial_pitch)
    assert abs(metrics["final_pitch_rad"]) < 0.025


@pytest.mark.parametrize("initial_roll", [-0.05, 0.05])
def test_roll_disturbance_is_recovered(classic_controller, initial_roll):
    metrics = classic_controller.run_rollout(duration=5.0, initial_roll=initial_roll)

    assert metrics["fell"] is False
    assert metrics["minimum_base_height_m"] > 0.1
    assert abs(metrics["final_roll_rad"]) < abs(initial_roll)


def test_classic_observation_builder_matches_61d_contract(classic_controller):
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    builder = classic_controller.ClassicObservationBuilder(
        model, data, np.zeros(model.nu, dtype=np.float32)
    )

    observation = builder.get(np.zeros(model.nu), np.zeros(13))

    assert observation.shape == (61,)
    assert np.isfinite(observation).all()


def test_distilled_feedback_artifact_generates_bounded_action(
    classic_controller, tmp_path
):
    harmonics = 2
    feature_count = classic_controller.OBSERVATION_DIM + 2 * harmonics
    weights = np.zeros((feature_count + 1, 14))
    weights[0] = 2.0
    artifact = tmp_path / "controller.npz"
    np.savez(
        artifact,
        format_version=np.array(1),
        observation_mean=np.zeros(feature_count),
        observation_scale=np.ones(feature_count),
        weights=weights,
        gait_frequency_hz=np.array(2.5),
        harmonics=np.array(harmonics),
        phase_offset_s=np.array(0.0),
    )

    controller = classic_controller.DistilledFeedbackController(artifact)
    action = controller.action(np.zeros(61), 0.0)

    assert action.shape == (14,)
    assert np.all(action == 1.0)


def test_distillation_fits_a_linear_feedback_mapping(distill_classic):
    rng = np.random.default_rng(7)
    observations = rng.normal(size=(80, 61)).astype(np.float32)
    times = np.arange(len(observations), dtype=np.float32) * 0.02
    expected_actions = observations[:, :14] * 0.1
    dataset = {
        "observations": observations,
        "actions": expected_actions,
        "times_s": times,
    }
    args = type(
        "FitArgs",
        (),
        {"gait_frequency": 2.5, "harmonics": 2, "ridge": 1e-5},
    )()

    mean, scale, weights, rmse = distill_classic._fit_controller(dataset, args)

    assert mean.shape == (65,)
    assert scale.shape == (65,)
    assert weights.shape == (66, 14)
    assert rmse < 0.001
