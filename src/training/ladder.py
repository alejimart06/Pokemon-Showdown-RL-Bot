"""
ladder.py
Entrenamiento/evaluacion contra el ladder real de Pokemon Showdown.

Se usa para:
  1. Refinar el modelo despues del self-play
  2. Evaluar el rendimiento real contra humanos
  3. Recopilar datos de partidas reales para analisis

Requisitos:
  - Cuenta en Pokemon Showdown (username/password en config.yaml)
  - Modelo pre-entrenado via self-play

Nota: el ladder tiene cooldowns entre partidas y puede ser lento.
      Usar principalmente para refinamiento y evaluacion, no para
      el grueso del entrenamiento.
"""

import asyncio
import os
import yaml
from poke_env import AccountConfiguration, ShowdownServerConfiguration

from src.bot.player import ShowdownPlayer
from src.agent.rl_agent import load_agent, build_callbacks


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def run_ladder(
    config_path: str = "config/config.yaml",
    model_path: str | None = None,
    n_battles: int = 100,
    train: bool = True,
):
    """
    Conecta el bot al ladder de Showdown y juega partidas reales.

    Args:
        config_path: ruta al archivo de configuracion
        model_path:  ruta al modelo pre-entrenado (obligatorio)
        n_battles:   numero de partidas a jugar
        train:       si True, el agente sigue aprendiendo durante las partidas
    """
    config = load_config(config_path)
    creds = config.get("credentials", {})
    reward_cfg = config.get("reward", {})
    battle_format = config.get("battle", {}).get("format", "gen9randombattle")
    model_dir = config.get("training", {}).get("model_dir", "models/")

    username = creds.get("username", "")
    password = creds.get("password", "")

    if not username or not password:
        raise ValueError(
            "Debes configurar username y password en config/config.yaml "
            "para jugar en el ladder."
        )

    if not model_path or not os.path.exists(model_path + ".zip"):
        raise ValueError(
            f"Modelo no encontrado en {model_path}. "
            "Entrena primero con self-play antes de ir al ladder."
        )

    account_cfg = AccountConfiguration(username, password)
    server_cfg = ShowdownServerConfiguration

    # Crear jugador con cuenta real
    player = ShowdownPlayer(
        reward_config=reward_cfg,
        battle_format=battle_format,
        account_configuration=account_cfg,
        server_configuration=server_cfg,
        start_challenging=False,
    )

    # Cargar modelo pre-entrenado
    agent = load_agent(model_path, player)
    callbacks = build_callbacks(config) if train else []

    print(f"[ladder] Conectando como '{username}'")
    print(f"[ladder] Formato: {battle_format}")
    print(f"[ladder] Partidas: {n_battles} | Entrenamiento activo: {train}")

    # Jugar en el ladder
    await player.ladder(n_battles)

    if train:
        final_path = os.path.join(model_dir, "ladder_refined_model")
        agent.save(final_path)
        print(f"[ladder] Modelo refinado guardado en {final_path}")

    # Estadisticas
    wins = player.n_won_battles
    total = player.n_finished_battles
    print(f"[ladder] Resultado: {wins}/{total} victorias ({100*wins/max(total,1):.1f}%)")
