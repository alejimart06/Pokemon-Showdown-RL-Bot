"""
self_play.py
Entrenamiento por self-play: el bot juega contra si mismo en un
servidor local de Pokemon Showdown.

Requisitos:
  - Servidor local corriendo:
    node pokemon-showdown start --no-security

Flujo:
  1. Crear el entorno ShowdownEnv (dos agentes internos que juegan entre si)
  2. Envolverlo con SingleAgentWrapper para stable-baselines3
  3. Entrenar PPO sobre el entorno
  4. Guardar checkpoints periodicamente
"""

import os
import yaml
from poke_env import LocalhostServerConfiguration

from src.bot.player import make_single_agent_env
from src.agent.rl_agent import build_agent, load_agent, build_callbacks


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_self_play(config_path: str = "config/config.yaml", resume: str | None = None):
    """
    Lanza el entrenamiento self-play.

    Args:
        config_path: ruta al archivo de configuracion
        resume:      ruta a modelo guardado para continuar entrenamiento
    """
    config = load_config(config_path)
    training_cfg = config.get("training", {})
    reward_cfg = config.get("reward", {})
    battle_format = config.get("battle", {}).get("format", "gen9randombattle")

    total_timesteps = training_cfg.get("total_timesteps", 1_000_000)
    model_dir = training_cfg.get("model_dir", "models/")

    os.makedirs(model_dir, exist_ok=True)

    # Crear entorno compatible con stable-baselines3
    env = make_single_agent_env(
        reward_config=reward_cfg,
        battle_format=battle_format,
        server_configuration=LocalhostServerConfiguration,
    )

    # Construir o cargar agente PPO
    if resume and os.path.exists(resume + ".zip"):
        print(f"[self_play] Cargando modelo desde {resume}")
        agent = load_agent(resume, env)
    else:
        print("[self_play] Creando agente nuevo")
        agent = build_agent(env, config)

    callbacks = build_callbacks(config)

    print(f"[self_play] Iniciando entrenamiento: {total_timesteps} timesteps")
    print(f"[self_play] Formato: {battle_format}")

    agent.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=(resume is None),
    )

    final_path = os.path.join(model_dir, "final_model")
    agent.save(final_path)
    print(f"[self_play] Modelo final guardado en {final_path}")
