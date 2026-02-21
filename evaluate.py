"""
evaluate.py
Evalua el modelo entrenado jugando N partidas contra SimpleHeuristicsPlayer
y muestra estadisticas detalladas: winrate, duracion media, rewards.

Uso:
  python evaluate.py                                        # usa models/final_model por defecto
  python evaluate.py --model models/self_play_model --battles 200
"""

import argparse
import asyncio
import yaml
import numpy as np
from poke_env import LocalhostServerConfiguration
from poke_env.player import SimpleHeuristicsPlayer, RandomPlayer
from stable_baselines3 import PPO

from src.bot.player import SelfPlayOpponent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluacion del bot")
    parser.add_argument("--model",   default="models/final_model", help="Ruta al modelo")
    parser.add_argument("--battles", type=int, default=100,         help="Numero de partidas")
    parser.add_argument("--opponent", choices=["heuristic", "random"], default="heuristic",
                        help="Tipo de oponente (default: heuristic)")
    parser.add_argument("--config",  default="config/config.yaml",  help="Ruta al config")
    return parser.parse_args()


async def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    battle_format = config.get("battle", {}).get("format", "gen9randombattle")
    server_cfg    = LocalhostServerConfiguration

    # Cargar modelo
    print(f"Cargando modelo: {args.model}")
    model = PPO.load(args.model)

    # Crear agente evaluador
    agent = SelfPlayOpponent(
        model=model,
        battle_format=battle_format,
        server_configuration=server_cfg,
    )

    # Crear oponente
    if args.opponent == "heuristic":
        opponent = SimpleHeuristicsPlayer(
            battle_format=battle_format,
            server_configuration=server_cfg,
        )
        opp_name = "SimpleHeuristicsPlayer"
    else:
        opponent = RandomPlayer(
            battle_format=battle_format,
            server_configuration=server_cfg,
        )
        opp_name = "RandomPlayer"

    print(f"Evaluando {args.battles} partidas contra {opp_name}...")
    print("-" * 45)

    await agent.battle_against(opponent, n_battles=args.battles)

    # Estadisticas
    n_won   = agent.n_won_battles
    n_total = agent.n_finished_battles
    winrate = 100 * n_won / max(n_total, 1)

    print(f"\n{'='*45}")
    print(f"  Resultados tras {n_total} partidas")
    print(f"{'='*45}")
    print(f"  Victorias : {n_won}")
    print(f"  Derrotas  : {n_total - n_won}")
    print(f"  Winrate   : {winrate:.1f}%")
    print(f"{'='*45}\n")

    if winrate >= 60:
        print("El bot supera claramente al heuristico. Listo para self-play.")
    elif winrate >= 50:
        print("El bot gana por poco. Puede mejorar con mas entrenamiento.")
    else:
        print("El bot pierde contra el heuristico. Necesita mas entrenamiento.")


if __name__ == "__main__":
    asyncio.run(main())
