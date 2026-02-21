"""
self_play.py
Entrenamiento por self-play: el bot juega contra si mismo en un
servidor local de Pokemon Showdown.

Modos:
  - vs_heuristic: el agente aprende contra SimpleHeuristicsPlayer (fase inicial)
  - vs_self:      el agente aprende contra una copia congelada de si mismo (self-play real)

En vs_self, cada `update_opponent_freq` timesteps el oponente actualiza
sus pesos con el modelo mas reciente del agente (curriculo progresivo).
Esto evita que el agente se estanque explotando sus propios bugs.
"""

import os
import yaml
from stable_baselines3.common.callbacks import BaseCallback
from poke_env import LocalhostServerConfiguration

from src.bot.player import make_single_agent_env, SelfPlayOpponent
from src.agent.rl_agent import build_agent, load_agent, build_callbacks


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_self_play(
    config_path: str = "config/config.yaml",
    resume: str | None = None,
    vs_self: bool = False,
):
    """
    Lanza el entrenamiento self-play.

    Args:
        config_path: ruta al archivo de configuracion
        resume:      ruta a modelo guardado para continuar entrenamiento
        vs_self:     True = self-play real (agente vs copia de si mismo)
                     False = agente vs SimpleHeuristicsPlayer (fase inicial)
    """
    config = load_config(config_path)
    training_cfg = config.get("training", {})
    reward_cfg = config.get("reward", {})
    battle_format = config.get("battle", {}).get("format", "gen9randombattle")

    total_timesteps = training_cfg.get("total_timesteps", 1_000_000)
    update_freq = training_cfg.get("update_opponent_freq", 50_000)

    # Subcarpeta distinta segun el modo de entrenamiento
    if vs_self:
        model_dir = training_cfg.get("model_dir_self", "models/vs_self/")
    else:
        model_dir = training_cfg.get("model_dir_heuristic", "models/vs_heuristic/")

    os.makedirs(model_dir, exist_ok=True)

    server_cfg = LocalhostServerConfiguration

    if vs_self:
        # --- Self-play real: oponente usa el modelo PPO ---
        if not resume:
            raise ValueError(
                "Para self-play necesitas un modelo base. "
                "Usa --resume models/final_model"
            )

        # Creamos primero el agente para poder pasarlo al oponente
        # El env se crea con un oponente placeholder que luego sustituimos
        print("[self_play] Modo: agente vs si mismo")

        # Oponente temporal para construir el env
        from poke_env.player import SimpleHeuristicsPlayer
        placeholder_opp = SimpleHeuristicsPlayer(
            battle_format=battle_format,
            server_configuration=server_cfg,
        )

        env = make_single_agent_env(
            reward_config=reward_cfg,
            battle_format=battle_format,
            server_configuration=server_cfg,
            opponent=placeholder_opp,
        )

        agent = load_agent(resume, env)

        # Crear oponente real con el modelo cargado
        opponent = SelfPlayOpponent(
            model=agent,
            battle_format=battle_format,
            server_configuration=server_cfg,
        )

        # Recrear env con el oponente real
        env = make_single_agent_env(
            reward_config=reward_cfg,
            battle_format=battle_format,
            server_configuration=server_cfg,
            opponent=opponent,
        )
        agent.set_env(env)

        # Callback que actualiza los pesos del oponente periodicamente
        callbacks = build_callbacks(config, model_dir) + [
            _UpdateOpponentCallback(opponent, agent, update_freq)
        ]

    else:
        # --- Fase inicial: oponente heuristico ---
        print("[self_play] Modo: agente vs heuristico")
        env = make_single_agent_env(
            reward_config=reward_cfg,
            battle_format=battle_format,
            server_configuration=server_cfg,
        )

        if resume and os.path.exists(resume + ".zip"):
            print(f"[self_play] Cargando modelo desde {resume}")
            agent = load_agent(resume, env)
        else:
            print("[self_play] Creando agente nuevo")
            agent = build_agent(env, config)

        callbacks = build_callbacks(config, model_dir)

    print(f"[self_play] Formato: {battle_format}")
    print(f"[self_play] Timesteps: {total_timesteps}")

    agent.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=(resume is None),
    )

    final_path = os.path.join(model_dir, "self_play_model" if vs_self else "final_model")
    agent.save(final_path)
    print(f"[self_play] Modelo guardado en {final_path}")


class _UpdateOpponentCallback(BaseCallback):
    """
    Callback de stable-baselines3 que actualiza los pesos del oponente
    cada N timesteps con el modelo actual del agente.
    """

    def __init__(self, opponent: SelfPlayOpponent, agent, update_freq: int):
        super().__init__(verbose=1)
        self._opponent = opponent
        self._agent = agent
        self._update_freq = update_freq
        self._last_update = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update >= self._update_freq:
            self._opponent.update_model(self._agent)
            self._last_update = self.num_timesteps
            if self.verbose:
                print(f"[self_play] Oponente actualizado en step {self.num_timesteps}")
        return True
