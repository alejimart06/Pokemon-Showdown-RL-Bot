"""
rl_agent.py
Wrapper del agente MaskablePPO de sb3-contrib.

Usamos MaskablePPO (en lugar de PPO normal) porque en Pokemon Showdown
no todas las acciones son válidas en cada turno:
  - No puedes cambiar a un pokemon fainted
  - No puedes usar un movimiento sin PP
  - No puedes mega/z/tera si no tienes esa opción disponible
  - Si te queda 1 pokemon, no puedes cambiar

Con PPO normal el agente elige acciones inválidas y poke-env hace fallback
aleatorio, lo que rompe el aprendizaje. MaskablePPO recibe una máscara
binaria por step y aplica -inf logit a las acciones inválidas, garantizando
que el agente NUNCA elija una acción ilegal.

La arquitectura de la red neuronal es una MLP estándar:
  obs_vector -> [256, 256] -> policy head / value head
"""

import os
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv


def build_agent(env: VecEnv, config: dict) -> MaskablePPO:
    """
    Construye un agente MaskablePPO nuevo.

    Args:
        env:    entorno vectorizado de gymnasium (debe exponer action_masks())
        config: diccionario con hiperparámetros (de config.yaml -> ppo)
    """
    ppo_cfg      = config.get("ppo", {})
    training_cfg = config.get("training", {})
    log_dir      = training_cfg.get("log_dir", "logs/")

    policy_kwargs = dict(
        net_arch=[256, 256],
    )

    agent = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        n_steps=ppo_cfg.get("n_steps", 2048),
        batch_size=ppo_cfg.get("batch_size", 64),
        n_epochs=ppo_cfg.get("n_epochs", 10),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        ent_coef=ppo_cfg.get("ent_coef", 0.01),
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        verbose=1,
    )

    return agent


def load_agent(path: str, env: VecEnv) -> MaskablePPO:
    """
    Carga un agente MaskablePPO previamente guardado.

    Args:
        path: ruta al archivo .zip del modelo (sin extensión)
        env:  entorno vectorizado (necesario para continuar entrenamiento)
    """
    agent = MaskablePPO.load(path, env=env)
    return agent


def build_callbacks(config: dict, model_dir: str | None = None) -> list:
    """
    Construye los callbacks de entrenamiento:
      - CheckpointCallback: guarda el modelo cada N steps

    Args:
        config:    configuración completa del proyecto
        model_dir: carpeta donde guardar checkpoints.
    """
    training_cfg = config.get("training", {})
    save_freq    = training_cfg.get("save_freq", 50_000)

    if model_dir is None:
        model_dir = training_cfg.get("model_dir", "models/")

    os.makedirs(model_dir, exist_ok=True)

    callbacks = [
        CheckpointCallback(
            save_freq=save_freq,
            save_path=model_dir,
            name_prefix="showdown_ppo",
            verbose=1,
        )
    ]

    return callbacks
