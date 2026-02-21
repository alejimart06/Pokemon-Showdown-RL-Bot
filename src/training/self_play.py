"""
self_play.py
Entrenamiento por self-play: el bot juega contra una copia de si mismo
en un servidor local de Pokemon Showdown.

Requisitos:
  - Tener el servidor local de Showdown corriendo (ver README)
  - poke-env gestiona automaticamente multiples instancias en paralelo

Flujo:
  1. Crear dos jugadores: agente (aprende) y oponente (copia congelada)
  2. El agente juega contra el oponente
  3. Cada N partidas, actualizar el oponente con los pesos actuales del agente
     (esto evita que el agente aprenda a explotar sus propios bugs)
  4. Guardar checkpoints periodicamente
"""

import asyncio
import os
import yaml
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env import LocalhostServerConfiguration

from src.bot.player import ShowdownPlayer
from src.agent.rl_agent import build_agent, load_agent, build_callbacks


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def run_self_play(config_path: str = "config/config.yaml", resume: str | None = None):
    """
    Lanza el entrenamiento self-play.

    Args:
        config_path: ruta al archivo de configuracion
        resume:      ruta a un modelo guardado para continuar entrenamiento
    """
    config = load_config(config_path)
    training_cfg = config.get("training", {})
    reward_cfg = config.get("reward", {})
    battle_format = config.get("battle", {}).get("format", "gen9randombattle")

    total_timesteps = training_cfg.get("total_timesteps", 1_000_000)
    n_envs = training_cfg.get("n_envs", 4)
    model_dir = training_cfg.get("model_dir", "models/")

    server_cfg = LocalhostServerConfiguration

    # --- Crear el agente jugador ---
    agent_player = ShowdownPlayer(
        reward_config=reward_cfg,
        battle_format=battle_format,
        server_configuration=server_cfg,
        start_challenging=False,
    )

    # --- Oponente inicial: heuristico simple para dar senales de reward claras ---
    # Mas adelante se reemplaza por una copia congelada del propio agente
    opponent = SimpleHeuristicsPlayer(
        battle_format=battle_format,
        server_configuration=server_cfg,
    )

    # --- Construir o cargar agente PPO ---
    env = agent_player
    if resume and os.path.exists(resume):
        print(f"[self_play] Cargando modelo desde {resume}")
        agent = load_agent(resume, env)
    else:
        print("[self_play] Creando agente nuevo")
        agent = build_agent(env, config)

    callbacks = build_callbacks(config)

    # --- Entrenamiento ---
    print(f"[self_play] Iniciando entrenamiento: {total_timesteps} timesteps")
    print(f"[self_play] Formato: {battle_format}")

    agent_player.play_against(
        env_algorithm=_train_ppo,
        opponent=opponent,
        env_algorithm_kwargs={
            "agent": agent,
            "total_timesteps": total_timesteps,
            "callbacks": callbacks,
            "model_dir": model_dir,
        },
    )

    # Guardar modelo final
    final_path = os.path.join(model_dir, "final_model")
    agent.save(final_path)
    print(f"[self_play] Modelo final guardado en {final_path}")


def _train_ppo(player: ShowdownPlayer, agent, total_timesteps: int, callbacks: list, model_dir: str):
    """Funcion de entrenamiento que se ejecuta dentro del contexto de poke-env."""
    agent.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=False,
    )
