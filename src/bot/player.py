"""
player.py
Clase principal del bot. Hereda de Gen9EnvPlayer de poke-env,
que implementa la interfaz de gymnasium (Environment de RL).

poke-env se encarga automaticamente de:
  - Conectarse al servidor de Showdown
  - Parsear todos los mensajes de la batalla en tiempo real
  - Exponer el estado del juego a traves de battle objects
  - Enviar las ordenes al servidor

Nosotros solo necesitamos definir:
  - calc_reward: cuanto reward dar en cada turno
  - embed_battle: como convertir el estado a vector numerico
  - describe_embedding: metadatos del espacio de observacion
"""

import numpy as np
from gymnasium.spaces import Box, Discrete
from poke_env.player import Gen9EnvPlayer
from poke_env.environment import AbstractBattle

from src.state.encoder import encode_battle, get_observation_size
from src.bot.action_space import ACTION_SPACE_SIZE, action_to_move, get_valid_action
from src.agent.reward import compute_reward


class ShowdownPlayer(Gen9EnvPlayer):
    """
    Bot de Pokemon Showdown que expone una interfaz gymnasium
    compatible con stable-baselines3.
    """

    def __init__(self, reward_config: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self._reward_config = reward_config or {}
        self._prev_own_fainted = 0
        self._prev_opp_fainted = 0

    # ------------------------------------------------------------------
    # Interfaz gymnasium requerida por poke-env
    # ------------------------------------------------------------------

    def calc_reward(self, last_battle: AbstractBattle, current_battle: AbstractBattle) -> float:
        return compute_reward(last_battle, current_battle, self._reward_config)

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return encode_battle(battle)

    def describe_embedding(self):
        obs_size = get_observation_size()
        return Box(
            low=np.full(obs_size, -1.0, dtype=np.float32),
            high=np.full(obs_size, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def action_space(self):
        return Discrete(ACTION_SPACE_SIZE)

    # ------------------------------------------------------------------
    # Conversion accion -> orden para poke-env
    # ------------------------------------------------------------------

    def action_to_move(self, action: int, battle: AbstractBattle):
        """
        Traduce el indice de accion del agente a la orden concreta.
        Si la accion no esta disponible, usa el fallback.
        """
        safe_action = get_valid_action(action, battle)
        order = action_to_move(safe_action, battle)

        if order is None:
            # Ultimo recurso: dejar que poke-env elija
            return self.choose_random_move(battle)

        if hasattr(order, "id"):
            # Es un Move
            return self.create_order(order)
        else:
            # Es un Pokemon (cambio)
            return self.create_order(order)
