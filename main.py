"""
main.py
Punto de entrada del bot de Pokemon Showdown con RL.

=============================================================================
FLUJO RECOMENDADO (nueva version v2 con encoder mejorado Gen 6+):
=============================================================================

  # FASE 1 — Entrenar v2 contra el bot heuristico desde cero
  python main.py --mode self_play --tag v2

  # FASE 1b — Continuar entrenamiento v2 vs heuristico
  python main.py --mode self_play --tag v2 --resume models/v2_vs_heuristic/final_model

  # FASE 2 — Self-play: v2 juega contra si mismo (necesita modelo base de fase 1)
  python main.py --mode self_play --vs-self --tag v2 --resume models/v2_vs_heuristic/final_model

  # Entrenar version original (sin tag = v1, carpetas originales)
  python main.py --mode self_play
  python main.py --mode self_play --vs-self --resume models/vs_heuristic/final_model

  # Evaluar en el ladder online
  python main.py --mode ladder --model models/v2_vs_self/self_play_model --battles 100

=============================================================================
EVALUACION Y JUEGO:
=============================================================================

  # IA vs IA: v2 contra el bot heuristico
  python battle_ai_vs_ai.py --model models/v2_vs_self/self_play_model --battles 100

  # IA vs IA: v2 contra v1 (benchmark)
  python battle_ai_vs_ai.py \\
      --model models/v2_vs_self/self_play_model \\
      --opponent-model models/vs_self/self_play_model \\
      --battles 200

  # Jugar contra el bot en el navegador (con selector de version)
  python play_vs_bot.py
  python play_vs_bot.py --model models/v2_vs_self/self_play_model --name BotV2
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bot de Pokemon Showdown con Reinforcement Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["self_play", "ladder"],
        required=True,
        help="Modo: self_play (entrenamiento local) | ladder (online)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Ruta al archivo de configuracion (default: config/config.yaml)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help=(
            "Etiqueta de version del modelo, p.ej. 'v2'. "
            "Guarda en models/<tag>_vs_heuristic/ y models/<tag>_vs_self/. "
            "Sin tag usa las carpetas originales (models/vs_heuristic/, models/vs_self/)."
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="[self_play] Ruta a modelo guardado para continuar entrenamiento (sin .zip)",
    )
    parser.add_argument(
        "--vs-self",
        action="store_true",
        help="[self_play] Self-play: el agente juega contra una copia de si mismo",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="[ladder] Ruta al modelo pre-entrenado para usar en el ladder (sin .zip)",
    )
    parser.add_argument(
        "--battles",
        type=int,
        default=100,
        help="[ladder] Numero de partidas a jugar (default: 100)",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="[ladder] Solo evaluar, no seguir entrenando durante las partidas",
    )
    parser.add_argument(
        "--opponent-model",
        default=None,
        help=(
            "[self_play --vs-self] Ruta a un modelo fijo como oponente (sin .zip). "
            "Si se indica, el oponente NO actualiza sus pesos durante el entrenamiento."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "self_play":
        from src.training.self_play import run_self_play
        run_self_play(
            config_path=args.config,
            resume=args.resume,
            vs_self=args.vs_self,
            tag=args.tag,
            opponent_model_path=args.opponent_model,
        )

    elif args.mode == "ladder":
        import asyncio
        from src.training.ladder import run_ladder
        asyncio.run(run_ladder(
            config_path=args.config,
            model_path=args.model,
            n_battles=args.battles,
            train=not args.no_train,
        ))


if __name__ == "__main__":
    main()
