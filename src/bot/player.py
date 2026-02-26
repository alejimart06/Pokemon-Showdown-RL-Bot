"""
player.py
Clase principal del bot. Hereda de SinglesEnv de poke-env 0.11+.

ACCIÓN MASKING
==============
El problema central de usar PPO normal con Pokemon es que el espacio de
acciones tiene 26 slots pero en cada turno solo un subconjunto es válido.
Con PPO normal el agente elige acciones ilegales → poke-env hace fallback
aleatorio → el agente nunca aprende la correspondencia acción↔efecto.

Solución: MaskablePPO (sb3-contrib) + ActionMasker wrapper.
  - get_action_mask(battle) devuelve un array bool[26]
  - ActionMasker envuelve el env y expone action_masks() por step
  - MaskablePPO aplica -inf logit a las acciones enmascaradas
  - El agente NUNCA puede elegir una acción ilegal

INCERTIDUMBRE SOBRE EL RIVAL
=============================
El encoder codifica lo conocido y marca con 0 lo desconocido:
  - Movimientos del rival: solo los vistos en batalla (resto = 0)
  - Item del rival: 0 si no se ha revelado
  - Habilidad del rival: 0 si no se ha revelado
  - Hazards: Stealth Rock, Spikes (1-3 capas), Toxic Spikes (1-2), Sticky Web
"""

import numpy as np
from gymnasium.spaces import Box
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from poke_env.environment import SinglesEnv, SingleAgentWrapper
from poke_env.player import Player, SimpleHeuristicsPlayer
from poke_env.player.battle_order import DefaultBattleOrder
from poke_env.battle import AbstractBattle

from src.bot.action_space import get_action_mask
from src.state.encoder import encode_battle, get_observation_size
from src.agent.reward import RewardTracker
from src.utils.battle_logger import BattleLogger


class ShowdownEnv(SinglesEnv):
    """
    Entorno PettingZoo de Pokemon Showdown.
    Hereda de SinglesEnv (poke-env 0.11+).

    strict=True: si el agente elige una acción inválida lanza error
    (con MaskablePPO esto nunca debería ocurrir).
    """

    def __init__(self, reward_config: dict | None = None,
                 log_dir: str = "logs/", **kwargs):
        # strict=True ahora que MaskablePPO garantiza acciones válidas
        kwargs.setdefault("strict", False)   # False como seguridad, mask lo garantiza
        super().__init__(**kwargs)
        self._tracker = RewardTracker(reward_config or {})
        self._training_logger = BattleLogger(
            log_dir=log_dir,
            report_every=1000,
            report_name="training_battle_report.md",
        )
        obs_size  = get_observation_size()
        obs_space = Box(
            low=np.full(obs_size, -1.0, dtype=np.float32),
            high=np.full(obs_size, 1.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_spaces = {
            agent: obs_space for agent in self.possible_agents
        }

    def calc_reward(self, battle: AbstractBattle) -> float:
        if battle.finished:
            self._training_logger.end_battle(battle)
            self._tracker.reset()
        return self._tracker.compute(battle)

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return encode_battle(battle)

    def action_to_order(self, action, battle, fake=False, strict=True):
        # Turno forzado: lista vacía o solo /choose default → devolver default
        valid = battle.valid_orders
        if len(valid) == 0 or (len(valid) == 1 and str(valid[0]) == '/choose default'):
            return DefaultBattleOrder()
        return SinglesEnv.action_to_order(action, battle, fake=fake, strict=strict)

    def order_to_action(self, order, battle, fake=False, strict=True):
        # Turno forzado: lista vacía o solo /choose default → devolver acción -2
        valid = battle.valid_orders
        if len(valid) == 0 or (len(valid) == 1 and str(valid[0]) == '/choose default'):
            return np.int64(-2)
        return SinglesEnv.order_to_action(order, battle, fake=fake, strict=strict)


def _mask_fn(env) -> np.ndarray:
    """
    Función de máscara compatible con ActionMasker de sb3-contrib.
    Devuelve un array bool[26] con las acciones válidas para este turno.

    La cadena de wrappers que construye SB3 es:
      DummyVecEnv → Monitor → ActionMasker → SingleAgentWrapper → ShowdownEnv

    _mask_fn recibe el env dentro de ActionMasker, que es SingleAgentWrapper.
    Necesitamos llegar al ShowdownEnv (PokeEnv) que tiene battle1.

    Recorremos la cadena .env hasta encontrar el atributo battle1.
    """
    node = env
    for _ in range(10):   # máximo 10 niveles de wrappers
        if hasattr(node, 'battle1'):
            battle = node.battle1
            if battle is None:
                return np.ones(26, dtype=bool)
            return np.array(get_action_mask(battle), dtype=bool)
        if hasattr(node, 'env'):
            node = node.env
        else:
            break
    # Fallback: sin batalla encontrada, permitir todo
    return np.ones(26, dtype=bool)


def make_single_agent_env(
    reward_config: dict | None = None,
    opponent=None,
    log_dir: str = "logs/",
    **kwargs,
) -> ActionMasker:
    """
    Crea el entorno listo para MaskablePPO:
      ShowdownEnv → SingleAgentWrapper → ActionMasker

    ActionMasker expone action_masks() en cada step para que
    MaskablePPO solo pueda elegir acciones válidas.

    Args:
        reward_config: coeficientes de reward (de config.yaml)
        opponent:      jugador oponente (Player de poke-env).
                       Si es None, se usa SimpleHeuristicsPlayer.
        log_dir:       directorio donde guardar training_battle_report.md
        **kwargs:      argumentos para ShowdownEnv.
    """
    env = ShowdownEnv(reward_config=reward_config, log_dir=log_dir, **kwargs)

    if opponent is None:
        battle_format = kwargs.get("battle_format", "gen9randombattle")
        server_cfg    = kwargs.get("server_configuration")
        opponent = SimpleHeuristicsPlayer(
            battle_format=battle_format,
            server_configuration=server_cfg,
        )

    wrapped = SingleAgentWrapper(env, opponent)
    masked  = ActionMasker(wrapped, _mask_fn)
    return masked


class SelfPlayOpponent(Player):
    """
    Oponente para self-play: usa un modelo MaskablePPO guardado.
    No aprende — solo actúa como oponente fijo.

    Cada cierto número de partidas, self_play.py actualiza sus pesos
    con el modelo más reciente del agente.
    """

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self._model = model

    def update_model(self, model):
        """Actualiza los pesos del oponente con el modelo más reciente."""
        self._model = model

    def choose_move(self, battle: AbstractBattle):
        # Turno de espera: el servidor solo acepta /choose default
        if getattr(battle, '_wait', False):
            return self.choose_default_move()
        # Sin opciones disponibles: también default
        if not battle.available_moves and not battle.available_switches:
            return self.choose_default_move()

        obs  = encode_battle(battle)
        mask = np.array(get_action_mask(battle), dtype=bool)

        # MaskablePPO acepta action_masks como kwarg en predict()
        action, _ = self._model.predict(obs, action_masks=mask, deterministic=True)
        return self._action_to_order(int(action), battle)

    def _action_to_order(self, action: int, battle: AbstractBattle):
        """
        Convierte el índice de acción en una orden de poke-env.
        Mismo mapeo que SinglesEnv.action_to_order para Gen9:
          0-5:   cambios
          6-9:   movimientos
          10-13: movimientos + mega
          14-17: movimientos + z-move
          18-21: movimientos + dynamax
          22-25: movimientos + tera
        """
        from poke_env.environment.singles_env import SinglesEnv as _SE
        try:
            return _SE.action_to_order(action, battle, fake=False, strict=False)
        except Exception:
            return self.choose_random_move(battle)
