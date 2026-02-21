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
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.battle import AbstractBattle

from src.state.encoder import encode_battle, get_observation_size
from src.agent.reward import RewardTracker


class ShowdownEnv(SinglesEnv):
    """
    Entorno PettingZoo de Pokemon Showdown.
    Hereda de SinglesEnv (poke-env 0.11+).
    """

    def __init__(self, reward_config: dict | None = None, **kwargs):
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
