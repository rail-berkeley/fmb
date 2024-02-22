"""Episode transforms for different RLDS datasets to canonical dataset definition."""
from typing import Any, Dict
from orca.data.rlds.fmb_primitive_peg_map import tf_primitive_to_primitive_id
import tensorflow as tf


def stanford_kuka_multimodal_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    # pad action to be 7-dimensional to fit Bridge data convention (add zero rotation action)
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def r2_d2_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["action"] = tf.concat(
        (
            trajectory["action_dict"]["cartesian_position"],
            trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


def fmb_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["ee_ft"] = tf.concat(
        (
            trajectory["observation"]["eef_force"],
            trajectory["observation"]["eef_torque"],
        ),
        axis=-1,
    )
    trajectory["observation"]["ee_vel"] =trajectory["observation"]["eef_vel"]
    trajectory["observation"]["ee_pose"] =trajectory["observation"]["eef_pose"]
    trajectory["observation"]["primitive_id"] = tf_primitive_to_primitive_id(trajectory["observation"]["primitive"])

    return trajectory


RLDS_TRAJECTORY_MAP_TRANSFORMS = {
    "stanford_kuka_multimodal_dataset": stanford_kuka_multimodal_dataset_transform,
    "r2_d2": r2_d2_dataset_transform,
    "r2_d2_pen": r2_d2_dataset_transform,
    "fmb_dataset": fmb_dataset_transform,
    "fmb_peg_dataset": fmb_dataset_transform,
    "fmb_peg4_dataset": fmb_dataset_transform,
    "fmb_peg6_dataset": fmb_dataset_transform,
    "fmb_peg8_dataset": fmb_dataset_transform,
}
