"""
battle_ai_vs_ai.py
Evaluacion: enfrenta dos bots entre si para medir el progreso del entrenamiento.

=============================================================================
MODOS DE USO
=============================================================================

  # 1. Bot v2 (nuevo) vs Bot Heuristico (SimpleHeuristicsPlayer)
  python battle_ai_vs_ai.py --model models/v2_vs_self/self_play_model --battles 100

  # 2. Bot v2 vs Bot v1 (benchmark version antigua vs nueva)
  python battle_ai_vs_ai.py \\
      --model models/v2_vs_self/self_play_model \\
      --opponent-model models/vs_self/self_play_model \\
      --battles 200

  # 3. Autodetectar ultima version entrenada en una carpeta
  python battle_ai_vs_ai.py --model-dir models/v2_vs_self/ --battles 100

  # 4. Benchmark completo: v2 vs Heuristico Y v2 vs v1
  python battle_ai_vs_ai.py --benchmark --tag v2 --battles 100

  # 5. Ver las formulas de calculo usadas por el bot
  python battle_ai_vs_ai.py --show-formulas

=============================================================================
"""

import asyncio
import argparse
import glob
import os
import time
from collections import defaultdict

from poke_env import LocalhostServerConfiguration
from poke_env.player import Player, SimpleHeuristicsPlayer
from poke_env.battle import AbstractBattle

from src.state.encoder import encode_battle


# ===========================================================================
# Agente RL
# ===========================================================================

class RLBotPlayer(Player):
    """Agente que usa un modelo MaskablePPO para elegir acciones."""

    def __init__(self, model, label: str = "RLBot", verbose: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._model   = model
        self._label   = label
        self._verbose = verbose

    def choose_move(self, battle: AbstractBattle):
        import numpy as np
        from src.bot.action_space import get_action_mask
        obs  = encode_battle(battle)
        mask = np.array(get_action_mask(battle), dtype=bool)
        action, _ = self._model.predict(obs, action_masks=mask, deterministic=True)
        if self._verbose:
            own = battle.active_pokemon
            opp = battle.opponent_active_pokemon
            if own and opp:
                print(f"  [{self._label}] {own.species} vs {opp.species} | "
                      f"HP {own.current_hp_fraction:.0%}/{opp.current_hp_fraction:.0%} | "
                      f"accion={int(action)}")
        return self._action_to_order(int(action), battle)

    def _action_to_order(self, action: int, battle: AbstractBattle):
        from poke_env.environment.singles_env import SinglesEnv as _SE
        try:
            return _SE.action_to_order(action, battle, fake=False, strict=False)
        except Exception:
            return self.choose_random_move(battle)


# ===========================================================================
# Estadisticas
# ===========================================================================

class BattleStats:
    def __init__(self, bot_label: str, opp_label: str):
        self.bot_label = bot_label
        self.opp_label = opp_label
        self.wins   = 0
        self.losses = 0
        self.ties   = 0
        self.turns  = []
        self.t0     = time.time()

    def record(self, won: bool, lost: bool, n_turns: int):
        if won:       self.wins   += 1
        elif lost:    self.losses += 1
        else:         self.ties   += 1
        self.turns.append(n_turns)

    @property
    def total(self):
        return self.wins + self.losses + self.ties

    @property
    def winrate(self):
        return self.wins / self.total * 100 if self.total else 0.0

    def print_live(self, i: int, n: int):
        status = "V" if (self.wins == i + 1) else ("D" if (self.losses == i - self.wins - self.ties + 1) else "E")
        # reconstruir estado de la ultima batalla
        last_win  = self.wins   > (self.total - 1 - self.ties - self.losses)
        last_lost = self.losses > (self.total - 1 - self.ties - self.wins)
        sym = "✓" if last_win else ("✗" if last_lost else "~")
        t = self.turns[-1] if self.turns else 0
        print(f"  [{sym}] Batalla {self.total:3d}/{n}  turnos={t:3d}  "
              f"winrate={self.winrate:.1f}%  ({self.wins}V {self.losses}D {self.ties}E)")

    def print_summary(self):
        elapsed  = time.time() - self.t0
        avg_t    = sum(self.turns) / len(self.turns) if self.turns else 0
        wr       = self.winrate

        print("\n" + "═" * 62)
        print(f"  RESULTADO  {self.bot_label}  vs  {self.opp_label}")
        print("═" * 62)
        print(f"  Partidas:          {self.total}")
        print(f"  Victorias:         {self.wins:3d}  ({wr:.1f}%)")
        print(f"  Derrotas:          {self.losses:3d}  ({self.losses/max(self.total,1)*100:.1f}%)")
        print(f"  Empates:           {self.ties:3d}  ({self.ties/max(self.total,1)*100:.1f}%)")
        print(f"  Turnos promedio:   {avg_t:.1f}")
        print(f"  Tiempo total:      {elapsed:.1f}s  ({elapsed/max(self.total,1):.2f}s/partida)")
        print("─" * 62)

        if wr >= 70:   verdict = "EXCELENTE  — domina claramente"
        elif wr >= 58: verdict = "BUENO      — ventaja consistente"
        elif wr >= 48: verdict = "EQUILIBRADO"
        elif wr >= 35: verdict = "MEJORABLE  — el oponente gana mas"
        else:          verdict = "NECESITA MAS ENTRENAMIENTO"
        print(f"  Veredicto: {verdict}")
        print("═" * 62)
        return wr


# ===========================================================================
# Buscar modelos en disco
# ===========================================================================

def _find_model(path_or_dir: str) -> str:
    """
    Resuelve la ruta al modelo:
    - Si termina en .zip o existe directamente, lo usa.
    - Si es un directorio, busca el checkpoint mas reciente.
    """
    if os.path.isfile(path_or_dir + ".zip") or os.path.isfile(path_or_dir):
        return path_or_dir

    if os.path.isdir(path_or_dir):
        candidates = (
            glob.glob(os.path.join(path_or_dir, "*.zip")) +
            glob.glob(os.path.join(path_or_dir, "*_model")) +
            glob.glob(os.path.join(path_or_dir, "final_model")) +
            glob.glob(os.path.join(path_or_dir, "self_play_model"))
        )
        if candidates:
            # el mas reciente por mtime
            best = sorted(candidates, key=os.path.getmtime)[-1]
            return best.replace(".zip", "")

    raise FileNotFoundError(
        f"No se encontro modelo en: {path_or_dir}\n"
        "Comprueba la ruta o entrena primero con:\n"
        "  python main.py --mode self_play --tag v2"
    )


def _list_available_models() -> list[tuple[str, str]]:
    """
    Devuelve lista de (label, ruta) de todos los modelos encontrados en models/.
    Ordena: primero self_play (fase 2), luego final_model (fase 1).
    """
    result = []
    if not os.path.isdir("models"):
        return result

    for entry in sorted(os.scandir("models"), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        for fname in ("self_play_model", "final_model"):
            fpath = os.path.join(entry.path, fname)
            if os.path.isfile(fpath + ".zip"):
                label = f"{entry.name}/{fname}"
                result.append((label, fpath))

    return result


# ===========================================================================
# Cargar modelo
# ===========================================================================

def _load_rl_model(model_path: str):
    from sb3_contrib import MaskablePPO
    print(f"  Cargando: {model_path}")
    return MaskablePPO.load(model_path)


# ===========================================================================
# Correr batallas
# ===========================================================================

async def run_battles(
    bot_path: str,
    opp_path: str | None,
    n_battles: int,
    battle_format: str,
    verbose: bool,
    bot_label: str = "Bot",
    opp_label: str | None = None,
    port: int = 8000,
) -> float:
    """
    Lanza N batallas. Devuelve el winrate del bot principal.
    """
    from poke_env import LocalhostServerConfiguration

    server_cfg = LocalhostServerConfiguration

    print(f"\n{'─'*62}")
    print(f"  Cargando modelos...")
    bot_model = _load_rl_model(bot_path)
    bot = RLBotPlayer(
        model=bot_model,
        label=bot_label,
        verbose=verbose,
        battle_format=battle_format,
        server_configuration=server_cfg,
        username=bot_label[:16],
    )

    if opp_path is not None:
        opp_model = _load_rl_model(opp_path)
        opp_label = opp_label or f"RL ({os.path.basename(opp_path)})"
        opponent = RLBotPlayer(
            model=opp_model,
            label=opp_label,
            verbose=False,
            battle_format=battle_format,
            server_configuration=server_cfg,
            username=(opp_label[:16] + "_opp"),
        )
    else:
        opp_label = opp_label or "SimpleHeuristicsPlayer"
        opponent = SimpleHeuristicsPlayer(
            battle_format=battle_format,
            server_configuration=server_cfg,
        )

    stats = BattleStats(bot_label, opp_label)
    print(f"  {bot_label}  vs  {opp_label}")
    print(f"  Formato: {battle_format}  |  {n_battles} batallas")
    print(f"{'─'*62}")

    for i in range(n_battles):
        await bot.battle_against(opponent, n_battles=1)

        all_battles = list(bot.battles.values())
        if all_battles:
            last = all_battles[-1]
            stats.record(
                won=bool(last.won),
                lost=bool(last.lost),
                n_turns=last.turn,
            )
            if verbose or (i + 1) % max(1, n_battles // 20) == 0:
                stats.print_live(i, n_battles)

    return stats.print_summary()


# ===========================================================================
# Benchmark completo (v2 vs Heuristico + v2 vs v1)
# ===========================================================================

async def run_benchmark(tag: str, n_battles: int, battle_format: str, verbose: bool):
    """
    Benchmark completo:
      1. <tag> vs SimpleHeuristicsPlayer
      2. <tag> vs version anterior (sin tag = v1)
    """
    tag_dir      = f"models/{tag}_vs_self/"
    v1_dir       = "models/vs_self/"
    fallback_tag = f"models/{tag}_vs_heuristic/"
    fallback_v1  = "models/vs_heuristic/"

    # Encontrar modelo del tag
    try:
        new_path = _find_model(tag_dir)
    except FileNotFoundError:
        try:
            new_path = _find_model(fallback_tag)
            print(f"[benchmark] Usando fase 1 ({fallback_tag}) porque fase 2 no existe aun.")
        except FileNotFoundError:
            print(f"[benchmark] ERROR: No se encontro modelo para tag='{tag}'.")
            print(f"  Entrena primero: python main.py --mode self_play --tag {tag}")
            return

    print(f"\n{'═'*62}")
    print(f"  BENCHMARK COMPLETO — tag: {tag}")
    print(f"{'═'*62}")

    # Test 1: nuevo vs heuristico
    print("\n[1/2] Nuevo bot vs SimpleHeuristicsPlayer")
    wr1 = await run_battles(
        bot_path=new_path,
        opp_path=None,
        n_battles=n_battles,
        battle_format=battle_format,
        verbose=verbose,
        bot_label=f"Bot-{tag}",
        opp_label="Heuristico",
    )

    # Test 2: nuevo vs v1 (si existe)
    try:
        v1_path = _find_model(v1_dir)
    except FileNotFoundError:
        try:
            v1_path = _find_model(fallback_v1)
        except FileNotFoundError:
            v1_path = None

    if v1_path:
        print(f"\n[2/2] Bot-{tag} vs Bot-v1")
        wr2 = await run_battles(
            bot_path=new_path,
            opp_path=v1_path,
            n_battles=n_battles,
            battle_format=battle_format,
            verbose=verbose,
            bot_label=f"Bot-{tag}",
            opp_label="Bot-v1",
        )
        print(f"\n{'═'*62}")
        print(f"  RESUMEN FINAL")
        print(f"  Bot-{tag} vs Heuristico: {wr1:.1f}% winrate")
        print(f"  Bot-{tag} vs Bot-v1:     {wr2:.1f}% winrate")
        mejora = wr2 - 50.0
        if mejora > 10:
            print(f"  Mejora clara sobre v1: +{mejora:.1f}pp sobre 50%")
        elif mejora > 0:
            print(f"  Ligera mejora sobre v1: +{mejora:.1f}pp")
        elif mejora < -5:
            print(f"  v1 sigue siendo mejor: {mejora:.1f}pp. Entrena mas o revisa el encoder.")
        else:
            print(f"  Equilibrado con v1 ({mejora:+.1f}pp). Sigue entrenando.")
        print(f"{'═'*62}")
    else:
        print("\n[2/2] No se encontro modelo v1. Omitiendo benchmark v1.")
        print("  Para entrenar v1: python main.py --mode self_play")


# ===========================================================================
# Mostrar formulas
# ===========================================================================

def show_formulas():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         FORMULAS Y CALCULOS USADOS EN EL BOT (Gen 6-9)         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  FORMULA DE DAÑO:                                                ║
║    base   = floor(floor(42 * power * atk/def / 50) + 2)         ║
║    damage = base * targets * weather * badge * critical          ║
║                  * random * STAB * type1 * type2                 ║
║                  * burn * other * zpower                         ║
║                                                                  ║
║  PARAMETROS (nivel 100, singles competitivo):                    ║
║    targets  = 1.0         badge   = 1.0  zpower = 1.0           ║
║    critical = 1.0 (prom)  random  = 0.925 (prom, 0.85-1.0)     ║
║    STAB     = 1.5  |  Adaptability: 2.0  |  Protean: siempre   ║
║    burn     = 0.5 (fisico quemado, Guts lo anula)               ║
║    other    = items + habilidades + pantallas + terreno          ║
║                                                                  ║
║  STATS (nivel 100, IVs 31, EVs 85):                              ║
║    stat = floor((2*base + 31 + 21)*100/100) + 5                  ║
║    HP   = floor((2*base + 31 + 21)*100/100) + 110                ║
║                                                                  ║
║  KO PROBABILISTICO:                                              ║
║    dmg_min (roll=0.85) >= HP  -> P_KO = 1.0 (garantizado)       ║
║    dmg_max (roll=1.00) <  HP  -> P_KO = 0.0 (imposible)         ║
║    entre medias -> interpolacion lineal                          ║
║                                                                  ║
║  CLIMA:   Sol: Fire*1.5 Water*0.5  |  Lluvia: Water*1.5 Fire*0.5║
║  TERRENO: Electric*1.3  Grass*1.3  Dragon*0.5  Psychic*1.3      ║
║  PANTALLAS (singles): Reflect/Light Screen/Aurora Veil: /2       ║
║                                                                  ║
║  ITEMS ATACANTE:                                                 ║
║    Life Orb *1.3  |  Choice Band/Specs *1.5  |  Expert Belt *1.2║
║    Muscle Band *1.1  |  Wise Glasses *1.1  |  Plates *1.2       ║
║                                                                  ║
║  ITEMS DEFENSOR:                                                 ║
║    Eviolite /1.5 (ambos) | Assault Vest /1.5 (especial)         ║
║    Berries de tipo /2 (Occa, Passho, Wacan, ... Roseli)         ║
║                                                                  ║
║  VELOCIDAD: stat_100(base) * boost * scarf(1.5) * abil          ║
║    Swift Swim/Chlorophyll/Sand Rush/Slush Rush *2 (clima)        ║
║    Surge Surfer *2 (Electric Terrain)                            ║
║    Quick Feet *1.5 (estado)  |  Unburden *2 (sin item)          ║
║                                                                  ║
║  HABILIDADES ATACANTE (seleccion):                               ║
║    Adaptability *2 STAB  | Technician *1.5 (pow<=60)            ║
║    Hustle *1.5 fis  | Guts *1.5 fis (estado, sin burn pen)      ║
║    Blaze/Torrent/Overgrow/Swarm *1.5 tipo (<1/3 HP)             ║
║    Sheer Force *1.3 | Iron Fist *1.2 | Strong Jaw *1.5          ║
║    Tough Claws *1.3 (contacto) | Mega Launcher *1.5 (pulso)     ║
║    Steelworker/Transistor/Dragon's Maw/Rocky Payload *1.5 tipo  ║
║    Aerilate/Pixilate/Refrigerate/Galvanize *1.2 (Normal->tipo)  ║
║    Analytic *1.3 (si va segundo)  | Neuroforce *1.25 (SE)       ║
║                                                                  ║
║  HABILIDADES DEFENSOR (seleccion):                               ║
║    Levitate/Flash Fire/Water Absorb/Volt Absorb/Sap Sipper:inmune║
║    Earth Eater (Ground) | Storm Drain (Water) | Well-Baked Body  ║
║    Thick Fat /2 (Fire,Ice) | Heatproof /2 (Fire)                ║
║    Ice Scales /2 (especial) | Multiscale/Shadow Shield /2 (HP lleno)║
║    Filter/Solid Rock/Prism Armor *0.75 (SE)                     ║
║    Fur Coat /2 (fisico) | Marvel Scale /1.5 fis (estado)        ║
║    Fluffy /2 contacto, *2 Fire | Purifying Salt /2 Ghost        ║
║    Bulletproof (balas/bombas) | Soundproof (sonido)             ║
║    Wonder Guard (solo SE)                                        ║
║                                                                  ║
║  Ver lista completa: src/state/items_and_abilities.py            ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluacion IA vs IA para el bot de Pokemon Showdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--model",
        default=None,
        help="Ruta al modelo principal a evaluar (sin .zip). Tambien acepta directorio.",
    )
    group.add_argument(
        "--benchmark",
        action="store_true",
        help="Modo benchmark completo: <tag> vs Heuristico y vs v1 (requiere --tag)",
    )
    group.add_argument(
        "--show-formulas",
        action="store_true",
        help="Mostrar tabla de formulas y salir",
    )

    parser.add_argument(
        "--opponent-model",
        default=None,
        help="Modelo oponente RL (sin .zip). Sin este flag usa SimpleHeuristicsPlayer.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Etiqueta de version, p.ej. 'v2'. Usado con --benchmark.",
    )
    parser.add_argument(
        "--battles",
        type=int,
        default=100,
        help="Numero de batallas (default: 100)",
    )
    parser.add_argument(
        "--format",
        default="gen9randombattle",
        help="Formato de batalla (default: gen9randombattle)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar detalles de cada turno",
    )
    parser.add_argument(
        "--bot-label",
        default=None,
        help="Nombre para mostrar del bot principal (default: nombre del archivo)",
    )
    parser.add_argument(
        "--opp-label",
        default=None,
        help="Nombre para mostrar del oponente",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.show_formulas:
        show_formulas()
        return

    if args.benchmark:
        if not args.tag:
            print("ERROR: --benchmark requiere --tag (p.ej. --tag v2)")
            return
        asyncio.run(run_benchmark(args.tag, args.battles, args.format, args.verbose))
        return

    if not args.model:
        # Sin argumentos: listar modelos disponibles
        print("\nModelos disponibles en models/:")
        models = _list_available_models()
        if not models:
            print("  (ninguno — entrena primero con: python main.py --mode self_play --tag v2)")
        else:
            for label, path in models:
                print(f"  {label}")
        print("\nUso: python battle_ai_vs_ai.py --model <ruta> [--opponent-model <ruta>] [--battles N]")
        return

    bot_path = _find_model(args.model)
    opp_path = _find_model(args.opponent_model) if args.opponent_model else None

    bot_label = args.bot_label or os.path.basename(bot_path)
    opp_label = args.opp_label or (
        os.path.basename(opp_path) if opp_path else "SimpleHeuristicsPlayer"
    )

    asyncio.run(run_battles(
        bot_path=bot_path,
        opp_path=opp_path,
        n_battles=args.battles,
        battle_format=args.format,
        verbose=args.verbose,
        bot_label=bot_label,
        opp_label=opp_label,
    ))


if __name__ == "__main__":
    main()
