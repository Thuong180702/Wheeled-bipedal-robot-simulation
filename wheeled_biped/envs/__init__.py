"""Environment modules - đăng ký các environment."""

from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.envs.base_env import WheeledBipedEnv
from wheeled_biped.envs.locomotion_env import LocomotionEnv
from wheeled_biped.envs.stair_env import StairEnv
from wheeled_biped.envs.standup_env import StandUpEnv
from wheeled_biped.envs.terrain_env import TerrainEnv
from wheeled_biped.envs.walking_env import WalkingEnv

# Registry cho dễ tra cứu
ENV_REGISTRY: dict[str, type] = {
    "BalanceEnv": BalanceEnv,
    "LocomotionEnv": LocomotionEnv,
    "WalkingEnv": WalkingEnv,
    "StairEnv": StairEnv,
    "TerrainEnv": TerrainEnv,
    "StandUpEnv": StandUpEnv,
}


def make_env(env_name: str, **kwargs) -> WheeledBipedEnv:
    """Tạo environment theo tên.

    Args:
        env_name: tên env (key trong ENV_REGISTRY).

    Returns:
        Instance của environment.
    """
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Env '{env_name}' không tồn tại. Có sẵn: {list(ENV_REGISTRY.keys())}")
    return ENV_REGISTRY[env_name](**kwargs)
