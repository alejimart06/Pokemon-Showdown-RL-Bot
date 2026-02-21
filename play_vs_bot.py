"""
play_vs_bot.py
Pone el bot a escuchar desafios en el servidor local.
Tu te conectas desde el navegador y lo desafias manualmente.

Pasos:
  1. Ejecuta este script
  2. Abre http://localhost:8000 en el navegador
  3. Entra como invitado con cualquier nombre
  4. Busca al usuario 'PokeBot' y desafialo a gen9randombattle

Uso:
  python play_vs_bot.py                              # usa models/final_model
  python play_vs_bot.py --model models/self_play_model
  python play_vs_bot.py --battles 5                 # acepta 5 desafios y para
"""

import argparse
import asyncio
import yaml
from poke_env import LocalhostServerConfiguration, AccountConfiguration
from stable_baselines3 import PPO

from src.bot.player import SelfPlayOpponent


def parse_args():
    parser = argparse.ArgumentParser(description="Juega contra el bot en el servidor local")
    parser.add_argument("--model",   default="models/final_model", help="Ruta al modelo")
    parser.add_argument("--battles", type=int, default=10,          help="Partidas a aceptar")
    parser.add_argument("--name",    default="PokeBot",             help="Nombre del bot en Showdown")
    parser.add_argument("--config",  default="config/config.yaml",  help="Ruta al config")
    return parser.parse_args()


async def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    battle_format = config.get("battle", {}).get("format", "gen9randombattle")

    print(f"Cargando modelo: {args.model}")
    model = PPO.load(args.model)

    # El bot se conecta al servidor local con el nombre elegido
    account_cfg = AccountConfiguration(args.name, None)
    server_cfg  = LocalhostServerConfiguration

    bot = SelfPlayOpponent(
        model=model,
        account_configuration=account_cfg,
        battle_format=battle_format,
        server_configuration=server_cfg,
    )

    print(f"\nBot '{args.name}' conectado al servidor local.")
    print(f"Abre http://localhost:8000 en el navegador.")
    print(f"Entra como invitado y desafia a '{args.name}'.")
    print(f"Formato: {battle_format}")
    print(f"Esperando {args.battles} desafio(s)... (Ctrl+C para salir)\n")

    await bot.accept_challenges(None, args.battles)

    n_won   = bot.n_won_battles
    n_total = bot.n_finished_battles
    print(f"\nResultado: {n_won}/{n_total} victorias del bot ({100*n_won/max(n_total,1):.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())
