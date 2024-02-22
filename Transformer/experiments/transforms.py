import tensorflow as tf
from typing import Any, Dict

def fmb_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["eef_vel"],
            trajectory["observation"]["eef_force"],
            trajectory["observation"]["eef_torque"],
        ),
        axis=-1,
    )
    return trajectory