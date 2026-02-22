"""
self_play.py
Entrenamiento por self-play: el bot juega contra si mismo en un
servidor local de Pokemon Showdown.

Modos:
  - vs_heuristic: el agente aprende contra SimpleHeuristicsPlayer (fase inicial)
  - vs_self:      el agente aprende contra una copia congelada de si mismo (self-play real)

El parametro `tag` permite entrenar distintas versiones sin sobreescribir
los modelos anteriores:
  - tag=None  -> modelos/vs_heuristic/  y  models/vs_self/  (compatibilidad v1)
  - tag="v2"  -> models/v2_vs_heuristic/  y  models/v2_vs_self/

En vs_self, cada `update_opponent_freq` timesteps el oponente actualiza
sus pesos con el modelo mas reciente del agente (curriculo progresivo).
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


def _model_dirs(training_cfg: dict, vs_self: bool, tag: str | None) -> str:
    """
    Devuelve la carpeta de modelos segun el modo y el tag de version.

    Sin tag:  usa las claves del config (retrocompatible con v1)
    Con tag:  models/<tag>_vs_heuristic/  o  models/<tag>_vs_self/
    """
    if tag:
        suffix = "vs_self" if vs_self else "vs_heuristic"
        return f"models/{tag}_{suffix}/"
    else:
        if vs_self:
            return training_cfg.get("model_dir_self", "models/vs_self/")
        else:
            return training_cfg.get("model_dir_heuristic", "models/vs_heuristic/")


def run_self_play(
    config_path: str = "config/config.yaml",
    resume: str | None = None,
    vs_self: bool = False,
    tag: str | None = None,
):
    """
    Lanza el entrenamiento self-play.

    Args:
        config_path: ruta al archivo de configuracion
        resume:      ruta a modelo guardado para continuar (sin .zip)
        vs_self:     True = self-play real (agente vs copia de si mismo)
        tag:         etiqueta de version, p.ej. 'v2'. Sin tag = v1 (retrocompatible)
    """
    config       = load_config(config_path)
    training_cfg = config.get("training", {})
    reward_cfg   = config.get("reward", {})
    battle_format = config.get("battle", {}).get("format", "gen9randombattle")

    total_timesteps = training_cfg.get("total_timesteps", 1_000_000)
    update_freq     = training_cfg.get("update_opponent_freq", 50_000)
    model_dir       = _model_dirs(training_cfg, vs_self, tag)

    os.makedirs(model_dir, exist_ok=True)
    server_cfg = LocalhostServerConfiguration

    version_label = f"[{tag}]" if tag else "[v1]"

    if vs_self:
        # ----------------------------------------------------------------
        # Fase 2: self-play real — el oponente usa el modelo PPO
        # ----------------------------------------------------------------
        if not resume:
            raise ValueError(
                "Para self-play necesitas un modelo base. Usa:\n"
                f"  --resume models/{tag + '_' if tag else ''}vs_heuristic/final_model"
            )

        print(f"{version_label} Modo: agente vs si mismo")

        from poke_env.player import SimpleHeuristicsPlayer
        # Placeholder para poder construir el env antes de tener el modelo
        placeholder_opp = SimpleHeuristicsPlayer(
            battle_format=battle_format,
            server_configuration=server_cfg,
        )

        env   = make_single_agent_env(
            reward_config=reward_cfg,
            battle_format=battle_format,
            server_configuration=server_cfg,
            opponent=placeholder_opp,
        )
        agent = load_agent(resume, env)

        # Crear oponente real con los pesos del modelo cargado
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

        callbacks = build_callbacks(config, model_dir) + [
            _UpdateOpponentCallback(opponent, agent, update_freq)
        ]

    else:
        # ----------------------------------------------------------------
        # Fase 1: bot heuristico como oponente
        # ----------------------------------------------------------------
        print(f"{version_label} Modo: agente vs heuristico")

        env = make_single_agent_env(
            reward_config=reward_cfg,
            battle_format=battle_format,
            server_configuration=server_cfg,
        )

        if resume and os.path.exists(resume + ".zip"):
            print(f"{version_label} Cargando modelo desde: {resume}")
            agent = load_agent(resume, env)
        else:
            print(f"{version_label} Creando agente nuevo")
            agent = build_agent(env, config)

        callbacks = build_callbacks(config, model_dir)

    print(f"{version_label} Formato:    {battle_format}")
    print(f"{version_label} Timesteps:  {total_timesteps:,}")
    print(f"{version_label} Guardando en: {model_dir}")

    agent.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=(resume is None),
        use_masking=True,   # MaskablePPO: nunca elige acciones invalidas
    )

    final_name = "self_play_model" if vs_self else "final_model"
    final_path = os.path.join(model_dir, final_name)
    agent.save(final_path)
    print(f"{version_label} Modelo final guardado en: {final_path}.zip")

    # Indicar el siguiente paso al usuario
    if not vs_self:
        print(f"\n{version_label} SIGUIENTE PASO — Self-play (fase 2):")
        tag_flag = f"--tag {tag} " if tag else ""
        print(f"  python main.py --mode self_play --vs-self {tag_flag}--resume {final_path}")


class _UpdateOpponentCallback(BaseCallback):
    """
    Actualiza los pesos del oponente cada N timesteps con el modelo
    actual del agente (curriculo progresivo para self-play).
    """

    def __init__(self, opponent: SelfPlayOpponent, agent, update_freq: int):
        super().__init__(verbose=1)
        self._opponent    = opponent
        self._agent       = agent
        self._update_freq = update_freq
        self._last_update = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update >= self._update_freq:
            self._opponent.update_model(self._agent)
            self._last_update = self.num_timesteps
            if self.verbose:
                print(f"[self_play] Oponente actualizado en step {self.num_timesteps:,}")
        return True
