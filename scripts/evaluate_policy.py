#!/usr/bin/env python3
"""Run a headless MuJoCo rollout and report locomotion metrics."""

import argparse
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np

import infer_policy

CONTROL_DECIMATION = 4
FALL_HEIGHT_M = 0.06


def _roll_pitch(quat):
    w, x, y, z = quat
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch_sin = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    return roll, math.asin(pitch_sin)


def evaluate(args):
    policy_path = Path(args.walking)
    if not policy_path.is_file():
        raise FileNotFoundError(f"Walking policy not found: {policy_path}")
    if args.duration <= 0:
        raise ValueError("--duration must be greater than zero")

    scene_path = Path(args.scene)
    model_dir = Path(infer_policy.__file__).resolve().parents[1]
    if not scene_path.is_absolute():
        scene_path = model_dir / scene_path

    bam_model = infer_policy.load_bam_model(
        infer_policy.BAM_KP_FW, args.vin, infer_policy.BAM_MAX_CURRENT
    )
    model, data, bam_ctrl, _ = infer_policy.load_mujoco_with_bam(
        str(scene_path), bam_model, 0.005, args.vin_drop_gain, infer_policy.BAM_VIN_MIN
    )
    policy = infer_policy.PolicyInference(
        model,
        data,
        walking_onnx_path=str(policy_path),
        bam_ctrl=bam_ctrl,
        new_cmd_obs=True,
        use_projected_gravity=True,
    )
    input_shape = policy.walking_session.get_inputs()[0].shape
    output_shape = policy.walking_session.get_outputs()[0].shape
    if input_shape[-1] != 61 or output_shape[-1] != 14:
        raise ValueError(
            f"Expected a 61-observation, 14-action policy; got {input_shape} -> {output_shape}"
        )

    policy.set_vel_cmd(args.lin_vel_x, args.lin_vel_y, args.ang_vel_z)
    freejoint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    qpos_adr = int(model.jnt_qposadr[freejoint_id])
    qvel_adr = int(model.jnt_dofadr[freejoint_id])
    actuator_qpos = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    data.qpos[qpos_adr : qpos_adr + 7] = [0.0, 0.0, 0.125, 1.0, 0.0, 0.0, 0.0]
    data.qpos[actuator_qpos] = policy.default_pose
    bam_ctrl.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)

    control_dt = CONTROL_DECIMATION * model.opt.timestep
    num_steps = max(1, round(args.duration / control_dt))
    forward_speeds = []
    lateral_speeds = []
    yaw_rates = []
    rolls = []
    pitches = []
    heights = []
    fell = False

    for _ in range(num_steps):
        action = policy.infer()
        if not np.isfinite(action).all():
            raise FloatingPointError("Policy produced a non-finite action")
        policy.apply_action(action)
        for _ in range(CONTROL_DECIMATION):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise FloatingPointError("MuJoCo state became non-finite")

        quat = data.qpos[qpos_adr + 3 : qpos_adr + 7]
        velocity_body = policy.quat_rotate_inverse(
            quat, data.qvel[qvel_adr : qvel_adr + 3]
        )
        roll, pitch = _roll_pitch(quat)
        height = float(data.qpos[qpos_adr + 2])
        forward_speeds.append(float(velocity_body[0]))
        lateral_speeds.append(float(velocity_body[1]))
        yaw_rates.append(float(data.qvel[qvel_adr + 5]))
        rolls.append(roll)
        pitches.append(pitch)
        heights.append(height)
        if height < FALL_HEIGHT_M:
            fell = True
            break

    window_steps = max(1, round(1.0 / control_dt))
    recent = slice(-window_steps, None)
    metrics = {
        "policy": str(policy_path),
        "command": {
            "vx_mps": float(policy.vel_cmd[0]),
            "vy_mps": float(policy.vel_cmd[1]),
            "yaw_rate_radps": float(policy.vel_cmd[2]),
        },
        "simulation_seconds": float(data.time),
        "control_steps": len(forward_speeds),
        "fell": fell,
        "final_base_height_m": heights[-1],
        "minimum_base_height_m": min(heights),
        "mean_forward_speed_last_second_mps": float(np.mean(forward_speeds[recent])),
        "mean_lateral_speed_last_second_mps": float(np.mean(lateral_speeds[recent])),
        "mean_yaw_rate_last_second_radps": float(np.mean(yaw_rates[recent])),
        "roll_rms_rad": float(np.sqrt(np.mean(np.square(rolls)))),
        "pitch_rms_rad": float(np.sqrt(np.mean(np.square(pitches)))),
        "maximum_tilt_rad": float(
            max(math.hypot(roll, pitch) for roll, pitch in zip(rolls, pitches))
        ),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walking", required=True, help="Path to a 61x14 ONNX walking policy")
    parser.add_argument("--scene", default=infer_policy.MICRODUCK_XML, help="MuJoCo scene XML")
    parser.add_argument("--lin-vel-x", type=float, default=0.3, help="Forward speed command (m/s)")
    parser.add_argument("--lin-vel-y", type=float, default=0.0, help="Lateral speed command (m/s)")
    parser.add_argument("--ang-vel-z", type=float, default=0.0, help="Yaw rate command (rad/s)")
    parser.add_argument("--duration", type=float, default=10.0, help="Maximum rollout duration (s)")
    parser.add_argument("--vin", type=float, default=7.4, help="BAM battery voltage (V)")
    parser.add_argument(
        "--vin-drop-gain", type=float, default=0.1, help="BAM load-dependent voltage sag gain"
    )
    parser.add_argument("--json-output", type=Path, help="Optional path to save the metrics JSON")
    args = parser.parse_args()

    metrics = evaluate(args)
    output = json.dumps(metrics, indent=2)
    print(output)
    if args.json_output:
        args.json_output.write_text(output + os.linesep, encoding="utf-8")


if __name__ == "__main__":
    main()
