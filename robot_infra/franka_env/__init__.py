from gymnasium.envs.registration import register
import numpy as np

register(
    id='Franka-FMB-v0',
    entry_point='franka_env.envs:FrankaFMB',
)
