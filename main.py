"""
main.py
Punto de entrada del bot de Pokemon Showdown con RL.

Uso:
  python main.py --mode self_play                  # Entrenar desde cero
  python main.py --mode self_play --resume models/showdown_ppo_50000_steps
  python main.py --mode ladder --model models/final_model --battles 100
  python main.py --mode ladder --model models/final_model --battles 50 --no-train
"""

import argparse
import asyncio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bot de Pokemon Showdown con Reinforcement Learning"
    )
    parser.add_argument(
        "--mode",
        choices=["self_play", "ladder"],
        required=True,
        help="Modo de entrenamiento: self_play (local) o ladder (online)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Ruta al archivo de configuracion (default: config/config.yaml)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="[self_play] Ruta a modelo guardado para continuar entrenamiento",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="[ladder] Ruta al modelo pre-entrenado para usar en el ladder",
    )
    parser.add_argument(
        "--battles",
        type=int,
        default=100,
        help="[ladder] Numero de partidas a jugar en el ladder (default: 100)",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="[ladder] Solo evaluar, no seguir entrenando durante las partidas",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    if args.mode == "self_play":
        from src.training.self_play import run_self_play
        await run_self_play(config_path=args.config, resume=args.resume)

    elif args.mode == "ladder":
        from src.training.ladder import run_ladder
        await run_ladder(
            config_path=args.config,
            model_path=args.model,
            n_battles=args.battles,
            train=not args.no_train,
        )


if __name__ == "__main__":
    asyncio.run(main())
