"""
player.py
Clase principal del bot. Hereda de SinglesEnv de poke-env 0.11+,
que implementa la interfaz PettingZoo (compatible con stable-baselines3
a traves de SingleAgentWrapper).

poke-env se encarga automaticamente de:
  - Conectarse al servidor de Showdown (local o real)
  - Parsear todos los mensajes de la batalla en tiempo real
  - Exponer el estado del juego a traves de battle objects
  - Enviar las ordenes al servidor

Nosotros solo necesitamos definir:
  - calc_reward: cuanto reward dar por cada estado
  - embed_battle: como convertir el estado a vector numerico
  - observation_spaces: metadatos del espacio de observacion
"""

import numpy as np
from gymnasium.spaces import Box
from poke_env.environment import SinglesEnv, SingleAgentWrapper
from poke_env.player import Player, SimpleHeuristicsPlayer
from poke_env.battle import AbstractBattle

from src.state.encoder import encode_battle, get_observation_size
from src.agent.reward import RewardTracker


class ShowdownEnv(SinglesEnv):
    """
    Entorno PettingZoo de Pokemon Showdown.
    Hereda de SinglesEnv (poke-env 0.11+).
    """

    def __init__(self, reward_config: dict | None = None, **kwargs):
        # strict=False: si el agente elige una accion invalida, poke-env
        # selecciona automaticamente una orden valida en lugar de lanzar error.
        # Esto es necesario durante el entrenamiento inicial donde el agente
        # aun no ha aprendido a respetar las restricciones del juego.
        kwargs.setdefault("strict", False)
        super().__init__(**kwargs)
        self._tracker = RewardTracker(reward_config or {})
        obs_size = get_observation_size()
        obs_space = Box(
            low=np.full(obs_size, -1.0, dtype=np.float32),
            high=np.full(obs_size, 1.0, dtype=np.float32),
            dtype=np.float32,
        )
        # observation_spaces es un dict agent_name -> Space
        self.observation_spaces = {
            agent: obs_space for agent in self.possible_agents
        }

    def calc_reward(self, battle: AbstractBattle) -> float:
        if battle.finished:
            self._tracker.reset()
        return self._tracker.compute(battle)

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return encode_battle(battle)


def make_single_agent_env(
    reward_config: dict | None = None,
    opponent=None,
    **kwargs,
) -> SingleAgentWrapper:
    """
    Crea el entorno envuelto en SingleAgentWrapper para usarlo
    directamente con stable-baselines3 (interfaz gymnasium estandar).

    Args:
        reward_config: coeficientes de reward (de config.yaml)
        opponent:      jugador oponente (Player de poke-env).
                       Si es None, se usa SimpleHeuristicsPlayer por defecto.
        **kwargs:      argumentos para ShowdownEnv (battle_format, server_configuration, etc.)
    """
    env = ShowdownEnv(reward_config=reward_config, **kwargs)

    if opponent is None:
        battle_format = kwargs.get("battle_format", "gen9randombattle")
        server_cfg = kwargs.get("server_configuration")
        opponent = SimpleHeuristicsPlayer(
            battle_format=battle_format,
            server_configuration=server_cfg,
        )

    return SingleAgentWrapper(env, opponent)


class SelfPlayOpponent(Player):
    """
    Oponente para self-play: usa un modelo PPO guardado para elegir acciones.
    No aprende — solo actua como oponente fijo con los pesos cargados.

    Cada cierto numero de partidas, self_play.py actualiza sus pesos
    con el modelo mas reciente del agente (curriculo progresivo).
    """

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self._model = model

    def update_model(self, model):
        """Actualiza los pesos del oponente con un modelo mas reciente."""
        self._model = model

    def choose_move(self, battle: AbstractBattle):
        obs = encode_battle(battle)
        action, _ = self._model.predict(obs, deterministic=True)
        return self._action_to_order(int(action), battle)

    def _action_to_order(self, action: int, battle: AbstractBattle):
        """
        Convierte el indice de accion en una orden de poke-env.
        Mismo mapeo que SinglesEnv.action_to_order para Gen9:
          0-5:   cambios
          6-9:   movimientos
          10-13: movimientos + mega
          14-17: movimientos + z-move
          18-21: movimientos + dynamax
          22-25: movimientos + tera
        """
        from poke_env.environment.singles_env import SinglesEnv as _SE
        from poke_env.battle import Battle

        # Reutilizamos la logica de poke-env directamente
        try:
            order = _SE.action_to_order(action, battle, fake=False, strict=False)
            return order
        except Exception:
            return self.choose_random_move(battle)
