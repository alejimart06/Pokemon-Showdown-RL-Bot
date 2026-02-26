"""
battle_logger.py
Registra el transcurso y estadisticas de las batallas del bot.

Cada `report_every` batallas reescribe logs/battle_report.md con:
  - Estadisticas agregadas del bloque de partidas
  - Transcripto turno a turno de las ultimas 10 batallas

Uso tipico:
    logger = BattleLogger("logs/", report_every=1000)

    # En choose_move():
    logger.log_turn(battle, action_type="move", action_name="flamethrower")

    # Cuando termina la batalla (on_battle_end):
    logger.end_battle(battle)

    # Al final de la sesion (aunque no se haya llegado a 1000):
    logger.force_report()
"""

import os
from datetime import datetime
from collections import Counter


# Movimientos de setup reconocidos (normalizados: minusculas, sin espacios ni guiones)
_SETUP_KEYWORDS = frozenset([
    'swordsdance', 'nastyplot', 'calmmind', 'dragondance',
    'quiverdance', 'shellsmash', 'coil', 'bulkup', 'workup',
    'honeclaws', 'sharpen', 'growth', 'acidarmor', 'irondefense',
    'barrier', 'amnesia', 'stockpile', 'cosmicpower', 'geomancy',
    'tidyup', 'victorydance', 'filletaway', 'terablast',
])


class TurnRecord:
    """Datos de un turno: que hizo el bot y como quedaron los HP."""

    __slots__ = (
        'turn_num', 'own_species', 'opp_species',
        'own_hp_before', 'opp_hp_before',
        'own_hp_after', 'opp_hp_after',
        'action_type', 'action_name',
    )

    def __init__(self, turn_num, own_species, opp_species,
                 own_hp_before, opp_hp_before, action_type, action_name):
        self.turn_num      = turn_num
        self.own_species   = own_species
        self.opp_species   = opp_species
        self.own_hp_before = own_hp_before
        self.opp_hp_before = opp_hp_before
        self.own_hp_after  = own_hp_before   # se actualiza al turno siguiente
        self.opp_hp_after  = opp_hp_before
        self.action_type   = action_type     # "move" | "switch"
        self.action_name   = action_name


class BattleRecord:
    """Resumen de una batalla completa."""

    def __init__(self, battle_id: int):
        self.battle_id     = battle_id
        self.turns         = []
        self.won           = None
        self.total_turns   = 0
        self.bot_fainted   = 0
        self.opp_fainted   = 0
        self.switches_made = 0
        self.moves_used    = Counter()
        self.setup_moves   = 0
        self.timestamp     = datetime.now().strftime("%Y-%m-%d %H:%M")

    def finalize(self, battle):
        """Rellena los campos de resultado a partir del objeto battle final."""
        self.won         = battle.won
        self.total_turns = battle.turn
        self.bot_fainted = sum(1 for p in battle.team.values() if p.fainted)
        self.opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)


class BattleLogger:
    """
    Registra batallas y escribe un informe Markdown cada `report_every` partidas.
    El archivo logs/battle_report.md se reescribe completamente cada vez.
    """

    def __init__(self, log_dir: str = "logs/", report_every: int = 1000,
                 report_name: str = "battle_report.md"):
        self.log_dir      = log_dir
        self.report_every = report_every
        self.report_name  = report_name
        os.makedirs(log_dir, exist_ok=True)

        self._total_battles = 0
        self._batch         = []    # bloque actual (hasta report_every batallas)
        self._recent        = []    # ultimas 10 batallas (para transcriptos)
        self._current       = None  # BattleRecord en curso

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def log_turn(self, battle, action_type: str, action_name: str):
        """
        Llamar en choose_move() justo antes de devolver la orden.

        action_type: "move" o "switch"
        action_name: nombre del movimiento (move.id) o especie del pokemon destino
        """
        if self._current is None:
            self._current = BattleRecord(self._total_battles + 1)

        own = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if own is None or opp is None:
            return

        # Actualizar los HP de salida del turno anterior
        if self._current.turns:
            prev = self._current.turns[-1]
            prev.own_hp_after = own.current_hp_fraction
            prev.opp_hp_after = opp.current_hp_fraction

        turn = TurnRecord(
            turn_num      = battle.turn,
            own_species   = getattr(own, 'species', '?'),
            opp_species   = getattr(opp, 'species', '?'),
            own_hp_before = own.current_hp_fraction,
            opp_hp_before = opp.current_hp_fraction,
            action_type   = action_type,
            action_name   = action_name,
        )
        self._current.turns.append(turn)

        # Estadisticas de comportamiento
        if action_type == "switch":
            self._current.switches_made += 1
        elif action_type == "move":
            self._current.moves_used[action_name] += 1
            norm = action_name.lower().replace(' ', '').replace('-', '').replace('_', '')
            if norm in _SETUP_KEYWORDS:
                self._current.setup_moves += 1

    def end_battle(self, battle):
        """
        Llamar cuando una batalla termina.
        En poke-env, sobreescribir on_battle_end() y llamar aqui.
        """
        if self._current is None:
            self._current = BattleRecord(self._total_battles + 1)

        rec = self._current
        rec.finalize(battle)

        # Actualizar HP final del ultimo turno
        own = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if rec.turns and own and opp:
            rec.turns[-1].own_hp_after = own.current_hp_fraction
            rec.turns[-1].opp_hp_after = opp.current_hp_fraction

        self._total_battles += 1
        self._batch.append(rec)
        self._recent.append(rec)
        if len(self._recent) > 10:
            self._recent.pop(0)

        self._current = None

        if len(self._batch) >= self.report_every:
            self._write_report()
            self._batch = []

    def force_report(self):
        """
        Escribe el informe con los datos acumulados hasta ahora,
        aunque no se haya llegado a report_every partidas.
        Util para llamar al final de cada sesion.
        """
        if self._batch:
            self._write_report()

    # ------------------------------------------------------------------
    # Generacion del informe Markdown
    # ------------------------------------------------------------------

    def _write_report(self):
        batch = self._batch
        if not batch:
            return

        n      = len(batch)
        wins   = sum(1 for b in batch if b.won is True)
        losses = sum(1 for b in batch if b.won is False)
        draws  = n - wins - losses

        avg_turns    = sum(b.total_turns   for b in batch) / n
        avg_switches = sum(b.switches_made for b in batch) / n
        avg_opp_ko   = sum(b.opp_fainted   for b in batch) / n
        avg_own_ko   = sum(b.bot_fainted   for b in batch) / n
        avg_setup    = sum(b.setup_moves   for b in batch) / n

        all_moves   = Counter()
        for b in batch:
            all_moves.update(b.moves_used)
        top_moves   = all_moves.most_common(10)
        total_moves = sum(all_moves.values())

        first_id = batch[0].battle_id
        last_id  = batch[-1].battle_id
        now      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"# Battle Report — Partidas {first_id}–{last_id}",
            f"",
            f"**Generado:** {now}  |  **Total historico de sesion:** {self._total_battles}",
            f"",
            f"---",
            f"",
            f"## Resumen ({n} partidas)",
            f"",
            f"| Metrica | Valor |",
            f"|---------|-------|",
            f"| Victorias  | {wins} / {n} ({wins/n*100:.1f}%) |",
            f"| Derrotas   | {losses} / {n} ({losses/n*100:.1f}%) |",
            f"| Empates    | {draws} |",
            f"| Turnos promedio | {avg_turns:.1f} |",
            f"",
            f"## Comportamiento del Bot",
            f"",
            f"| Metrica | Media por partida |",
            f"|---------|------------------|",
            f"| Cambios de Pokemon      | {avg_switches:.2f} |",
            f"| Movimientos de setup    | {avg_setup:.2f} |",
            f"| Rivales eliminados (KO) | {avg_opp_ko:.2f} / 6 |",
            f"| Propios perdidos        | {avg_own_ko:.2f} / 6 |",
            f"",
            f"## Movimientos mas usados (top 10)",
            f"",
            f"| # | Movimiento | Usos | % del total |",
            f"|---|-----------|------|-------------|",
        ]

        for i, (mv, count) in enumerate(top_moves, 1):
            pct = count / total_moves * 100 if total_moves > 0 else 0.0
            lines.append(f"| {i} | {mv} | {count} | {pct:.1f}% |")

        lines += [
            f"",
            f"---",
            f"",
            f"## Transcriptos — Ultimas {len(self._recent)} Batallas",
            f"",
        ]

        for rec in reversed(self._recent):
            result = "Victoria" if rec.won else ("Derrota" if rec.won is False else "Empate")
            lines += [
                f"### Batalla #{rec.battle_id} — {result} en {rec.total_turns} turnos  "
                f"({rec.timestamp})",
                f"",
                f"Rivales KO: **{rec.opp_fainted}/6**  |  "
                f"Propios perdidos: **{rec.bot_fainted}/6**  |  "
                f"Cambios: **{rec.switches_made}**  |  "
                f"Setup: **{rec.setup_moves}**",
                f"",
                f"| Turno | Bot | Rival | Tipo | Accion | HP Bot | HP Rival |",
                f"|-------|-----|-------|------|--------|--------|----------|",
            ]
            for t in rec.turns:
                own_hp   = f"{t.own_hp_before*100:.0f}%→{t.own_hp_after*100:.0f}%"
                opp_hp   = f"{t.opp_hp_before*100:.0f}%→{t.opp_hp_after*100:.0f}%"
                tipo_str = "cambio" if t.action_type == "switch" else "ataque"
                lines.append(
                    f"| {t.turn_num} | {t.own_species} | {t.opp_species} "
                    f"| {tipo_str} | {t.action_name} | {own_hp} | {opp_hp} |"
                )
            lines.append("")

        path = os.path.join(self.log_dir, self.report_name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(
            f"[BattleLogger] Informe guardado: {path}  "
            f"({n} partidas, {wins/n*100:.1f}% victorias del bot)"
        )
