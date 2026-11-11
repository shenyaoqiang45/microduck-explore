#!/usr/bin/env python3
"""Run a pure classical-feedback balance controller in headless MuJoCo."""

import argparse
import json
import math
import os
import threading
import time
from pathlib import Path

import infer_policy
import mujoco
import mujoco.viewer
import numpy as np

CONTROL_DECIMATION = 4
FALL_HEIGHT_M = 0.06
OBSERVATION_DIM = 61


def _roll_pitch(quat):
    w, x, y, z = quat
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.atan2(2 * (w * y - z * x), 1 - 2 * (y * y + z * z))
    return roll, pitch


def _initial_quaternion(roll, pitch):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return np.array([cr * cp, sr * cp, cr * sp, -sr * sp])


class ClassicBalanceController:
    """PD attitude feedback mapped to the ankle-pitch and hip-roll joints."""

    def __init__(
        self,
        default_pose,
        *,
        pitch_kp=4.0,
        pitch_kd=0.4,
        roll_kp=0.8,
        roll_kd=0.2,
        max_pitch_correction=0.35,
        max_roll_correction=0.2,
    ):
        self.default_pose = np.asarray(default_pose, dtype=float)
        self.pitch_kp = pitch_kp
        self.pitch_kd = pitch_kd
        self.roll_kp = roll_kp
        self.roll_kd = roll_kd
        self.max_pitch_correction = max_pitch_correction
        self.max_roll_correction = max_roll_correction

    def targets(self, data, qvel_adr):
        quat = data.qpos[self.qpos_adr + 3 : self.qpos_adr + 7]
        roll, pitch = _roll_pitch(quat)
        angular_velocity = data.qvel[qvel_adr + 3 : qvel_adr + 6]
        pitch_correction = float(
            np.clip(
                self.pitch_kp * pitch + self.pitch_kd * angular_velocity[1],
                -self.max_pitch_correction,
                self.max_pitch_correction,
            )
        )
        roll_correction = float(
            np.clip(
                self.roll_kp * roll + self.roll_kd * angular_velocity[0],
                -self.max_roll_correction,
                self.max_roll_correction,
            )
        )

        targets = self.default_pose.copy()
        targets[1] += roll_correction
        targets[10] -= roll_correction
        targets[4] += pitch_correction
        targets[13] -= pitch_correction
        return targets

    def reset(self, qpos_adr):
        self.qpos_adr = qpos_adr


class ClassicObservationBuilder:
    """Build the shared proprioception and command observation without a policy."""

    def __init__(self, model, data, default_pose):
        self.model = model
        self.data = data
        self.default_pose = np.asarray(default_pose, dtype=np.float32)
        freejoint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
        )
        trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        if min(freejoint_id, trunk_id, gyro_id) < 0:
            raise ValueError("Classic observation requires trunk, freejoint, and IMU gyro")
        self.trunk_id = trunk_id
        self.gyro_adr = int(model.sensor_adr[gyro_id])
        self.joint_qpos = model.jnt_qposadr[model.actuator_trnid[:, 0]]
        self.joint_qvel = model.jnt_dofadr[model.actuator_trnid[:, 0]]

    def get(self, last_action, command):
        if len(last_action) != self.model.nu:
            raise ValueError(f"Expected {self.model.nu} previous actions")
        if len(command) != 13:
            raise ValueError("Expected the 13-dimensional unified command")

        quat = self.data.xquat[self.trunk_id]
        gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        projected_gravity = _quat_rotate_inverse(quat, gravity)
        angular_velocity = self.data.sensordata[self.gyro_adr : self.gyro_adr + 3]
        joint_position = self.data.qpos[self.joint_qpos] - self.default_pose
        joint_velocity = self.data.qvel[self.joint_qvel]
        observation = np.concatenate(
            (
                angular_velocity,
                projected_gravity,
                joint_position,
                joint_velocity,
                last_action,
                command,
            )
        ).astype(np.float32)
        if observation.shape != (OBSERVATION_DIM,):
            raise ValueError(f"Expected {OBSERVATION_DIM} observations, got {observation.shape}")
        return observation


def _quat_rotate_inverse(quat, vector):
    w = quat[0]
    xyz = quat[1:4]
    cross = np.cross(xyz, vector) * 2
    return vector - w * cross + np.cross(xyz, cross)


class DistilledFeedbackController:
    """Linear state feedback plus an explicit periodic gait reference."""

    def __init__(self, model_path):
        with np.load(model_path, allow_pickle=False) as artifact:
            version = int(artifact["format_version"])
            self.observation_mean = artifact["observation_mean"].astype(float)
            self.observation_scale = artifact["observation_scale"].astype(float)
            self.weights = artifact["weights"].astype(float)
            self.gait_frequency_hz = float(artifact["gait_frequency_hz"])
            self.harmonics = int(artifact["harmonics"])
            self.phase_offset_s = float(artifact["phase_offset_s"])
        if version != 1:
            raise ValueError(f"Unsupported distilled-controller format: {version}")
        feature_count = OBSERVATION_DIM + 2 * self.harmonics
        if (
            self.harmonics < 1
            or self.observation_mean.shape != (feature_count,)
            or self.observation_scale.shape != (feature_count,)
            or self.weights.shape != (feature_count + 1, 14)
            or not np.isfinite(self.observation_mean).all()
            or not np.isfinite(self.observation_scale).all()
            or (self.observation_scale <= 0).any()
            or not np.isfinite(self.weights).all()
            or not math.isfinite(self.gait_frequency_hz)
            or self.gait_frequency_hz <= 0
        ):
            raise ValueError("Invalid distilled classic-controller artifact")

    def action(self, observation, time_s):
        phase = 2 * math.pi * self.gait_frequency_hz * (time_s + self.phase_offset_s)
        gait_features = np.array(
            [
                value
                for harmonic in range(1, self.harmonics + 1)
                for value in (math.sin(harmonic * phase), math.cos(harmonic * phase))
            ]
        )
        features = np.concatenate((observation, gait_features))
        normalized = (features - self.observation_mean) / self.observation_scale
        action = self.weights[0] + normalized @ self.weights[1:]
        return np.clip(action, -1.0, 1.0)


def run_rollout(
    *,
    duration=10.0,
    initial_roll=0.0,
    initial_pitch=0.0,
    scene=infer_policy.MICRODUCK_XML,
    vin=7.4,
    vin_drop_gain=0.1,
    controller_path=None,
    command=(0.0, 0.0, 0.0),
    render=False,
):
    if duration <= 0:
        raise ValueError("duration must be greater than zero")

    scene_path = Path(scene)
    if not scene_path.is_absolute():
        scene_path = Path(infer_policy.__file__).resolve().parents[1] / scene_path

    bam_model = infer_policy.load_bam_model(
        infer_policy.BAM_KP_FW, vin, infer_policy.BAM_MAX_CURRENT
    )
    model, data, bam_ctrl, _ = infer_policy.load_mujoco_with_bam(
        str(scene_path), bam_model, 0.005, vin_drop_gain, infer_policy.BAM_VIN_MIN
    )
    freejoint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    qpos_adr = int(model.jnt_qposadr[freejoint_id])
    qvel_adr = int(model.jnt_dofadr[freejoint_id])
    actuator_qpos = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    if len(command) != 3:
        raise ValueError("command must contain forward, lateral, and yaw-rate values")
    default_pose = infer_policy.DEFAULT_POSE[: model.nu]
    controller = (
        DistilledFeedbackController(controller_path)
        if controller_path is not None
        else ClassicBalanceController(default_pose)
    )
    if isinstance(controller, ClassicBalanceController):
        controller.reset(qpos_adr)
    observation_builder = (
        ClassicObservationBuilder(model, data, default_pose)
        if isinstance(controller, DistilledFeedbackController)
        else None
    )

    command_obs = np.zeros(13, dtype=np.float32)
    command_obs[:3] = command
    last_action = np.zeros(model.nu, dtype=np.float32)
    control_dt = CONTROL_DECIMATION * model.opt.timestep
    num_steps = max(1, round(duration / control_dt))
    rolls = []
    pitches = []
    heights = []
    forward_speeds = []
    lateral_speeds = []
    yaw_rates = []
    fell = False
    restart_requested = threading.Event()

    def reset_rollout():
        nonlocal last_action, fell
        mujoco.mj_resetData(model, data)
        data.qpos[qpos_adr : qpos_adr + 7] = [
            0.0,
            0.0,
            0.125,
            *_initial_quaternion(initial_roll, initial_pitch),
        ]
        data.qpos[actuator_qpos] = default_pose
        bam_ctrl.reset(data.qpos)
        bam_ctrl.q_target[:] = default_pose
        mujoco.mj_forward(model, data)
        last_action = np.zeros(model.nu, dtype=np.float32)
        rolls.clear()
        pitches.clear()
        heights.clear()
        forward_speeds.clear()
        lateral_speeds.clear()
        yaw_rates.clear()
        fell = False

    def key_callback(key):
        if key in (ord("R"), ord("r")):
            restart_requested.set()

    def step():
        nonlocal last_action, fell
        if isinstance(controller, DistilledFeedbackController):
            observation = observation_builder.get(last_action, command_obs)
            action = controller.action(observation, data.time)
            last_action = action.astype(np.float32)
            targets = default_pose + action
        else:
            targets = controller.targets(data, qvel_adr)
        bam_ctrl.q_target[:] = targets
        for _ in range(CONTROL_DECIMATION):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise FloatingPointError("MuJoCo state became non-finite")

        quat = data.qpos[qpos_adr + 3 : qpos_adr + 7]
        roll, pitch = _roll_pitch(quat)
        velocity = _quat_rotate_inverse(quat, data.qvel[qvel_adr : qvel_adr + 3])
        rolls.append(roll)
        pitches.append(pitch)
        heights.append(float(data.qpos[qpos_adr + 2]))
        forward_speeds.append(float(velocity[0]))
        lateral_speeds.append(float(velocity[1]))
        yaw_rates.append(float(data.qvel[qvel_adr + 5]))
        if heights[-1] < FALL_HEIGHT_M:
            fell = True

    reset_rollout()
    if render:
        with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback
        ) as viewer:
            viewer.sync()
            while viewer.is_running():
                deadline = time.monotonic()
                for _ in range(num_steps):
                    if not viewer.is_running() or restart_requested.is_set():
                        break
                    step()
                    viewer.sync()
                    if fell:
                        print("Controller fell; press R to replay or close the viewer.")
                        break
                    deadline += control_dt
                    time.sleep(max(0.0, deadline - time.monotonic()))
                while viewer.is_running() and not restart_requested.is_set():
                    viewer.sync()
                    time.sleep(0.02)
                if restart_requested.is_set():
                    restart_requested.clear()
                    reset_rollout()
    else:
        for _ in range(num_steps):
            step()
            if fell:
                break

    return {
        "controller": (
            "distilled_linear_feedback_with_gait_reference"
            if isinstance(controller, DistilledFeedbackController)
            else "classical_attitude_feedback"
        ),
        "policy_weights_loaded": False,
        "command": {
            "vx_mps": float(command[0]),
            "vy_mps": float(command[1]),
            "yaw_rate_radps": float(command[2]),
        },
        "simulation_seconds": float(data.time),
        "control_steps": len(heights),
        "fell": fell,
        "final_base_height_m": heights[-1],
        "minimum_base_height_m": min(heights),
        "mean_forward_speed_mps": float(np.mean(forward_speeds)),
        "mean_forward_speed_last_second_mps": float(
            np.mean(forward_speeds[-max(1, round(1.0 / control_dt)) :])
        ),
        "mean_lateral_speed_last_second_mps": float(
            np.mean(lateral_speeds[-max(1, round(1.0 / control_dt)) :])
        ),
        "mean_yaw_rate_last_second_radps": float(
            np.mean(yaw_rates[-max(1, round(1.0 / control_dt)) :])
        ),
        "initial_roll_rad": float(rolls[0]),
        "final_roll_rad": float(rolls[-1]),
        "initial_pitch_rad": float(pitches[0]),
        "final_pitch_rad": float(pitches[-1]),
        "roll_rms_rad": float(np.sqrt(np.mean(np.square(rolls)))),
        "pitch_rms_rad": float(np.sqrt(np.mean(np.square(pitches)))),
        "maximum_tilt_rad": float(
            max(math.hypot(roll, pitch) for roll, pitch in zip(rolls, pitches))
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=infer_policy.MICRODUCK_XML, help="MuJoCo scene XML")
    parser.add_argument("--duration", type=float, default=10.0, help="Rollout duration (s)")
    parser.add_argument("--initial-roll", type=float, default=0.0, help="Initial roll (rad)")
    parser.add_argument("--initial-pitch", type=float, default=0.0, help="Initial pitch (rad)")
    parser.add_argument(
        "--distilled-controller",
        type=Path,
        help="Run a distilled linear-feedback controller artifact instead of balance PD",
    )
    parser.add_argument("--viewer", action="store_true", help="Open the real-time MuJoCo viewer")
    parser.add_argument("--lin-vel-x", type=float, default=0.0, help="Forward command (m/s)")
    parser.add_argument("--lin-vel-y", type=float, default=0.0, help="Lateral command (m/s)")
    parser.add_argument("--ang-vel-z", type=float, default=0.0, help="Yaw command (rad/s)")
    parser.add_argument("--vin", type=float, default=7.4, help="BAM battery voltage (V)")
    parser.add_argument(
        "--vin-drop-gain", type=float, default=0.1, help="BAM load-dependent voltage sag gain"
    )
    parser.add_argument("--json-output", type=Path, help="Optional path to save metrics JSON")
    args = parser.parse_args()

    metrics = run_rollout(
        duration=args.duration,
        initial_roll=args.initial_roll,
        initial_pitch=args.initial_pitch,
        scene=args.scene,
        vin=args.vin,
        vin_drop_gain=args.vin_drop_gain,
        controller_path=args.distilled_controller,
        command=(args.lin_vel_x, args.lin_vel_y, args.ang_vel_z),
        render=args.viewer,
    )
    output = json.dumps(metrics, indent=2)
    print(output)
    if args.json_output:
        args.json_output.write_text(output + os.linesep, encoding="utf-8")


if __name__ == "__main__":
    main()
