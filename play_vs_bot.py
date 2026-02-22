"""
play_vs_bot.py
Juega contra el bot en el navegador (servidor local de Showdown).

=============================================================================
PASOS PARA JUGAR
=============================================================================
  1. Asegurate de que el servidor local esta activo (localhost:8000)
  2. Ejecuta este script (te preguntara contra que version jugar)
  3. Abre http://localhost:8000 en el navegador
  4. Entra como invitado con cualquier nombre
  5. Busca al usuario 'PokeBot' (o el nombre elegido) y desafialo

=============================================================================
FORMAS DE USO
=============================================================================

  # Selector interactivo de version (recomendado)
  python play_vs_bot.py

  # Especificar modelo directamente
  python play_vs_bot.py --model models/v2_vs_self/self_play_model

  # Jugar contra v1 con nombre personalizado
  python play_vs_bot.py --model models/vs_self/self_play_model --name BotV1

  # Aceptar N partidas y parar
  python play_vs_bot.py --battles 3

  # Mostrar los movimientos elegidos por el bot
  python play_vs_bot.py --verbose
"""

import argparse
import asyncio
import glob
import os
import yaml

from poke_env import LocalhostServerConfiguration, AccountConfiguration
from sb3_contrib import MaskablePPO
import numpy as np

from src.state.encoder import encode_battle
from src.bot.action_space import get_action_mask
from poke_env.player import Player
from poke_env.battle import AbstractBattle


# ===========================================================================
# Agente RL con verbose
# ===========================================================================

class RLBotPlayer(Player):
    """Agente que usa un modelo MaskablePPO para elegir acciones."""

    def __init__(self, model, label: str = "PokeBot", verbose: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._model   = model
        self._label   = label
        self._verbose = verbose
        self._turn_log = []

    def choose_move(self, battle: AbstractBattle):
        obs  = encode_battle(battle)
        mask = np.array(get_action_mask(battle), dtype=bool)
        action, _ = self._model.predict(obs, action_masks=mask, deterministic=True)
        action = int(action)

        if self._verbose:
            own = battle.active_pokemon
            opp = battle.opponent_active_pokemon
            if own and opp:
                # Determinar tipo de accion
                if action < 6:
                    action_str = f"Cambio a slot {action}"
                elif action < 10:
                    moves = list(own.moves.values())
                    mv = moves[action - 6] if action - 6 < len(moves) else None
                    action_str = f"Movimiento: {mv.id if mv else '?'}"
                else:
                    action_str = f"Accion especial {action}"

                print(f"  [{self._label} T{battle.turn}] "
                      f"{own.species}({own.current_hp_fraction:.0%}) "
                      f"vs {opp.species}({opp.current_hp_fraction:.0%}) "
                      f"-> {action_str}")

        return self._action_to_order(action, battle)

    def _action_to_order(self, action: int, battle: AbstractBattle):
        from poke_env.environment.singles_env import SinglesEnv as _SE
        try:
            return _SE.action_to_order(action, battle, fake=False, strict=False)
        except Exception:
            return self.choose_random_move(battle)


# ===========================================================================
# Buscar modelos disponibles
# ===========================================================================

def _find_available_models() -> list[tuple[str, str]]:
    """
    Devuelve lista de (descripcion, ruta) de todos los modelos en models/.
    Ordena priorizando self_play (fase 2) sobre final_model (fase 1).
    """
    result = []
    if not os.path.isdir("models"):
        return result

    priority = {"self_play_model": 0, "final_model": 1}

    entries = []
    for entry in os.scandir("models"):
        if not entry.is_dir():
            continue
        for fname in ("self_play_model", "final_model"):
            fpath = os.path.join(entry.path, fname)
            if os.path.isfile(fpath + ".zip"):
                # Generar descripcion humana
                dir_name = entry.name
                if "v2" in dir_name and "self" in dir_name:
                    desc = f"[NUEVO] Bot v2 — fase 2 self-play  ({dir_name})"
                elif "v2" in dir_name and "heuristic" in dir_name:
                    desc = f"[NUEVO] Bot v2 — fase 1 vs heuristico  ({dir_name})"
                elif "self" in dir_name and "vs" in dir_name:
                    desc = f"[v1] Bot original — fase 2 self-play  ({dir_name})"
                elif "heuristic" in dir_name:
                    desc = f"[v1] Bot original — fase 1 vs heuristico  ({dir_name})"
                else:
                    desc = f"{dir_name}/{fname}"
                entries.append((priority.get(fname, 9), desc, fpath))

    # Ordenar: v2 primero, self_play antes que final_model
    entries.sort(key=lambda x: (0 if "v2" in x[1].lower() else 1, x[0]))
    result = [(desc, path) for _, desc, path in entries]
    return result


def _selector_interactivo() -> str | None:
    """
    Muestra un menu numerado con los modelos disponibles.
    Devuelve la ruta elegida o None si no hay modelos.
    """
    modelos = _find_available_models()

    if not modelos:
        print("\n⚠  No se encontraron modelos entrenados en models/")
        print("   Entrena primero con:")
        print("     python main.py --mode self_play --tag v2")
        return None

    print("\n" + "═" * 62)
    print("  SELECTOR DE VERSION — ¿Contra que bot quieres jugar?")
    print("═" * 62)
    for i, (desc, _) in enumerate(modelos, 1):
        print(f"  [{i}] {desc}")
    print(f"  [0] Salir")
    print("─" * 62)

    while True:
        try:
            choice = input("  Elige una opcion: ").strip()
            if choice == "0":
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(modelos):
                _, path = modelos[idx]
                print(f"\n  Elegido: {modelos[idx][0]}")
                return path
            else:
                print(f"  Opcion invalida. Elige entre 1 y {len(modelos)} (o 0 para salir).")
        except (ValueError, KeyboardInterrupt):
            return None


# ===========================================================================
# Main
# ===========================================================================

async def main_async(model_path: str, bot_name: str, n_battles: int,
                     battle_format: str, verbose: bool):
    """Conecta el bot al servidor y acepta desafios."""
    print(f"\n  Cargando modelo: {model_path}")
    model = MaskablePPO.load(model_path)

    account_cfg = AccountConfiguration(bot_name, None)
    server_cfg  = LocalhostServerConfiguration

    bot = RLBotPlayer(
        model=model,
        label=bot_name,
        verbose=verbose,
        account_configuration=account_cfg,
        battle_format=battle_format,
        server_configuration=server_cfg,
    )

    print(f"\n{'═' * 62}")
    print(f"  Bot '{bot_name}' conectado.")
    print(f"  Abre http://localhost:8000 en el navegador.")
    print(f"  Entra como invitado y desafia a '{bot_name}'.")
    print(f"  Formato: {battle_format}")
    if verbose:
        print(f"  Modo verbose: se mostraran los movimientos del bot.")
    print(f"  Esperando {n_battles} desafio(s)... (Ctrl+C para salir)")
    print(f"{'═' * 62}\n")

    await bot.accept_challenges(None, n_battles)

    # Resultado final
    n_won   = bot.n_won_battles
    n_total = bot.n_finished_battles
    wr      = 100 * n_won / max(n_total, 1)

    print(f"\n{'═' * 62}")
    print(f"  RESULTADO FINAL ({bot_name})")
    print(f"  Partidas jugadas: {n_total}")
    print(f"  Victorias del bot: {n_won}  ({wr:.0f}%)")
    if wr >= 60:
        print(f"  El bot te ha ganado bien. ¡Sigue intentandolo!")
    elif wr >= 40:
        print(f"  Partidas muy igualadas.")
    else:
        print(f"  Te has impuesto al bot. ¡Buen trabajo!")
    print(f"{'═' * 62}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Juega contra el bot en el servidor local de Pokemon Showdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Ruta al modelo (sin .zip). "
            "Si no se especifica, se muestra un selector interactivo."
        ),
    )
    parser.add_argument(
        "--name",
        default="PokeBot",
        help="Nombre del bot en Showdown (default: PokeBot)",
    )
    parser.add_argument(
        "--battles",
        type=int,
        default=10,
        help="Numero de desafios a aceptar (default: 10)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Ruta al config (default: config/config.yaml)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar que movimiento elige el bot en cada turno",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Listar modelos disponibles y salir",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Leer formato del config
    battle_format = "gen9randombattle"
    if os.path.isfile(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
            battle_format = cfg.get("battle", {}).get("format", battle_format)

    # --list: solo mostrar modelos
    if args.list:
        modelos = _find_available_models()
        print("\nModelos disponibles:")
        if modelos:
            for desc, path in modelos:
                print(f"  {desc}")
                print(f"    -> {path}.zip")
        else:
            print("  (ninguno)")
        return

    # Elegir modelo
    if args.model:
        model_path = args.model
    else:
        model_path = _selector_interactivo()
        if model_path is None:
            return

    # Verificar que existe
    if not os.path.isfile(model_path + ".zip") and not os.path.isfile(model_path):
        print(f"\n  ERROR: No se encontro el modelo: {model_path}")
        print(f"  Comprueba la ruta o entrena con: python main.py --mode self_play --tag v2")
        return

    asyncio.run(main_async(
        model_path=model_path,
        bot_name=args.name,
        n_battles=args.battles,
        battle_format=battle_format,
        verbose=args.verbose,
    ))


if __name__ == "__main__":
    main()
