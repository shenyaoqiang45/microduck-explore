#!/usr/bin/env python3
"""Distill a walking teacher into linear feedback plus a periodic gait reference."""

import argparse
import json
import os
from pathlib import Path

import classic_controller
import infer_policy
import mujoco
import numpy as np

CONTROL_DECIMATION = 4
FALL_HEIGHT_M = 0.06


def _collect_teacher(args):
    teacher_path = Path(args.teacher)
    if not teacher_path.is_file():
        raise FileNotFoundError(f"Teacher policy not found: {teacher_path}")
    scene_path = Path(args.scene)
    if not scene_path.is_absolute():
        scene_path = Path(infer_policy.__file__).resolve().parents[1] / scene_path

    bam_model = infer_policy.load_bam_model(
        infer_policy.BAM_KP_FW, args.vin, infer_policy.BAM_MAX_CURRENT
    )
    model, data, bam_ctrl, _ = infer_policy.load_mujoco_with_bam(
        str(scene_path), bam_model, 0.005, args.vin_drop_gain, infer_policy.BAM_VIN_MIN
    )
    teacher = infer_policy.PolicyInference(
        model,
        data,
        walking_onnx_path=str(teacher_path),
        bam_ctrl=bam_ctrl,
        new_cmd_obs=True,
        use_projected_gravity=True,
    )
    input_shape = teacher.walking_session.get_inputs()[0].shape
    output_shape = teacher.walking_session.get_outputs()[0].shape
    if input_shape[-1] != classic_controller.OBSERVATION_DIM or output_shape[-1] != model.nu:
        raise ValueError(f"Unexpected teacher contract: {input_shape} -> {output_shape}")
    if model.nu != 14:
        raise ValueError(f"Expected 14 actuators, got {model.nu}")

    teacher.set_vel_cmd(args.lin_vel_x, args.lin_vel_y, args.ang_vel_z)
    freejoint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    qpos_adr = int(model.jnt_qposadr[freejoint_id])
    qvel_adr = int(model.jnt_dofadr[freejoint_id])
    actuator_qpos = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    data.qpos[qpos_adr : qpos_adr + 7] = [0.0, 0.0, 0.125, 1.0, 0.0, 0.0, 0.0]
    data.qpos[actuator_qpos] = teacher.default_pose
    bam_ctrl.reset(data.qpos)
    teacher.set_position_targets(teacher.default_pose)
    mujoco.mj_forward(model, data)

    observations = []
    actions = []
    forward_speeds = []
    heights = []
    control_dt = CONTROL_DECIMATION * model.opt.timestep
    num_steps = max(1, round(args.duration / control_dt))
    for _ in range(num_steps):
        observations.append(teacher.get_observations().copy())
        action = teacher.infer()
        if not np.isfinite(action).all():
            raise FloatingPointError("Teacher produced a non-finite action")
        actions.append(action.copy())
        teacher.apply_action(action)
        for _ in range(CONTROL_DECIMATION):
            bam_ctrl.update()
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise FloatingPointError("MuJoCo state became non-finite during teacher rollout")
        quat = data.qpos[qpos_adr + 3 : qpos_adr + 7]
        velocity = infer_policy.PolicyInference.quat_rotate_inverse(
            None, quat, data.qvel[qvel_adr : qvel_adr + 3]
        )
        forward_speeds.append(float(velocity[0]))
        heights.append(float(data.qpos[qpos_adr + 2]))
        if heights[-1] < FALL_HEIGHT_M:
            break

    window_steps = max(1, round(1.0 / control_dt))
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "times_s": np.arange(len(actions), dtype=np.float32) * control_dt,
        "command": np.array([args.lin_vel_x, args.lin_vel_y, args.ang_vel_z], dtype=np.float32),
        "teacher": str(teacher_path.resolve()),
        "teacher_metrics": {
            "simulation_seconds": float(data.time),
            "fell": min(heights) < FALL_HEIGHT_M,
            "minimum_base_height_m": min(heights),
            "mean_forward_speed_last_second_mps": float(
                np.mean(forward_speeds[-window_steps:])
            ),
        },
    }


def _features(observations, times_s, frequency_hz, harmonics):
    phase = 2 * np.pi * frequency_hz * times_s
    gait_features = np.stack(
        [
            feature
            for harmonic in range(1, harmonics + 1)
            for feature in (np.sin(harmonic * phase), np.cos(harmonic * phase))
        ],
        axis=1,
    )
    return np.concatenate((observations, gait_features), axis=1)


def _fit_controller(dataset, args):
    features = _features(
        dataset["observations"], dataset["times_s"], args.gait_frequency, args.harmonics
    )
    actions = dataset["actions"]
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-5)
    normalized = (features - mean) / scale
    design = np.column_stack((np.ones(len(normalized)), normalized))
    regularizer = np.eye(design.shape[1]) * args.ridge
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(
        design.T @ design + regularizer, design.T @ actions
    )
    predictions = design @ weights
    action_rmse = float(np.sqrt(np.mean(np.square(predictions - actions))))
    return mean, scale, weights, action_rmse


def _save_artifact(path, dataset, args, mean, scale, weights, action_rmse):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            format_version=np.array(1),
            observation_mean=mean,
            observation_scale=scale,
            weights=weights,
            gait_frequency_hz=np.array(args.gait_frequency),
            harmonics=np.array(args.harmonics),
            phase_offset_s=np.array(args.phase_offset),
            action_rmse=np.array(action_rmse),
            teacher=dataset["teacher"],
            training_command=dataset["command"],
            walking_validated=np.array(False),
        )


def distill(args):
    if args.duration <= 0 or args.validation_duration <= 0:
        raise ValueError("Rollout durations must be greater than zero")
    if args.gait_frequency <= 0 or args.harmonics < 1 or args.ridge <= 0:
        raise ValueError("Gait frequency, harmonics, and ridge must be positive")

    dataset = _collect_teacher(args)
    mean, scale, weights, action_rmse = _fit_controller(dataset, args)
    output_path = Path(args.output)
    _save_artifact(output_path, dataset, args, mean, scale, weights, action_rmse)

    if args.dataset_output:
        dataset_path = Path(args.dataset_output)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dataset_path,
            observations=dataset["observations"],
            actions=dataset["actions"],
            times_s=dataset["times_s"],
            command=dataset["command"],
        )

    validation = classic_controller.run_rollout(
        duration=args.validation_duration,
        scene=args.scene,
        vin=args.vin,
        vin_drop_gain=args.vin_drop_gain,
        controller_path=output_path,
        command=(args.lin_vel_x, args.lin_vel_y, args.ang_vel_z),
    )
    teacher_speed = dataset["teacher_metrics"]["mean_forward_speed_last_second_mps"]
    speed_error = abs(validation["mean_forward_speed_last_second_mps"] - teacher_speed)
    walking_validated = (
        not validation["fell"]
        and validation["minimum_base_height_m"] > 0.1
        and speed_error <= args.max_teacher_speed_error
    )
    with np.load(output_path, allow_pickle=False) as artifact:
        model_data = {key: artifact[key] for key in artifact.files}
    model_data["walking_validated"] = np.array(walking_validated)
    model_data["validation_speed_error_mps"] = np.array(speed_error)
    with output_path.open("wb") as stream:
        np.savez_compressed(stream, **model_data)

    return {
        "controller": "linear_feedback_with_periodic_gait_reference",
        "teacher_policy_used_for_offline_distillation_only": dataset["teacher"],
        "runtime_requires_teacher_policy": False,
        "training_samples": len(dataset["actions"]),
        "teacher_metrics": dataset["teacher_metrics"],
        "imitation_action_rmse": action_rmse,
        "candidate_controller": str(output_path.resolve()),
        "validation": validation,
        "teacher_speed_error_mps": speed_error,
        "walking_validated": walking_validated,
    }


def main():
    default_policy_dir = Path(
        os.environ.get(
            "LOCALAPPDATA", Path.home() / "AppData" / "Local"
        )
    ) / "microduck" / "policies"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher",
        type=Path,
        default=default_policy_dir / "alpha_walking.onnx",
        help="Teacher ONNX used only during data collection",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output linear-controller NPZ"
    )
    parser.add_argument("--dataset-output", type=Path, help="Optional teacher dataset NPZ")
    parser.add_argument("--scene", default=infer_policy.MICRODUCK_XML, help="MuJoCo scene XML")
    parser.add_argument("--lin-vel-x", type=float, default=0.3, help="Forward command (m/s)")
    parser.add_argument("--lin-vel-y", type=float, default=0.0, help="Lateral command (m/s)")
    parser.add_argument("--ang-vel-z", type=float, default=0.0, help="Yaw command (rad/s)")
    parser.add_argument("--duration", type=float, default=10.0, help="Teacher rollout duration")
    parser.add_argument(
        "--validation-duration", type=float, default=10.0, help="Student validation duration"
    )
    parser.add_argument("--gait-frequency", type=float, default=2.5, help="Gait frequency (Hz)")
    parser.add_argument("--harmonics", type=int, default=4, help="Fourier gait harmonics")
    parser.add_argument("--phase-offset", type=float, default=0.0, help="Gait phase offset (s)")
    parser.add_argument("--ridge", type=float, default=0.01, help="Ridge regularization")
    parser.add_argument(
        "--max-teacher-speed-error",
        type=float,
        default=0.05,
        help="Maximum speed difference from teacher for walking validation (m/s)",
    )
    parser.add_argument("--vin", type=float, default=7.4, help="BAM battery voltage (V)")
    parser.add_argument(
        "--vin-drop-gain", type=float, default=0.1, help="BAM load-dependent voltage sag gain"
    )
    args = parser.parse_args()
    result = distill(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
