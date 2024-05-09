from gym.envs.registration import register

register(
    id='Franka-FMB-v0',
    entry_point='envs.franka_fmb_env:FrankaFMB',
)
