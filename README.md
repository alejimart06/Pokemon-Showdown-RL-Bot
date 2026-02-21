# Pokemon Showdown RL Bot

A Reinforcement Learning bot for Pokemon Showdown that learns to play autonomously by analyzing battles in real time — no manual input required.

Built with [poke-env](https://github.com/hsahovic/poke-env) and [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) (PPO).

---

## Features

- **Fully automatic** — the bot reads and reacts to every battle event on its own (moves, switches, weather, terrain, status effects, stat boosts, etc.)
- **Reinforcement Learning (PPO)** — learns by playing, no hand-crafted heuristics needed
- **Hybrid training** — start with fast self-play on a local server, then refine on the real Showdown ladder
- **Modular design** — easy to extend the state encoder, reward function, or swap the RL algorithm

---

## Project Structure

```
proyecto1/
├── config/
│   └── config.yaml          # Server settings, credentials, hyperparameters
├── src/
│   ├── agent/
│   │   ├── rl_agent.py      # PPO agent (stable-baselines3)
│   │   └── reward.py        # Reward function
│   ├── bot/
│   │   ├── player.py        # poke-env player (connects to Showdown)
│   │   └── action_space.py  # Action space definition and masking
│   ├── state/
│   │   └── encoder.py       # Converts battle state to numeric vector
│   └── training/
│       ├── self_play.py     # Local self-play training
│       └── ladder.py        # Ladder (online) training/evaluation
├── models/                  # Saved model checkpoints
├── logs/                    # TensorBoard logs
├── requirements.txt
└── main.py                  # Entry point
```

---

## Requirements

- Python 3.10+
- Node.js 18+ (for the local Showdown server)
- Git

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd proyecto1
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up the local Pokemon Showdown server

The local server is required for self-play training. It lets the bot play thousands of games without internet or rate limits.

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
node pokemon-showdown start --no-security
```

Leave this terminal running. The server will be available at `localhost:8000`.

---

## Usage

### Self-play training (start here)

```bash
python main.py --mode self_play
```

Resume from a checkpoint:

```bash
python main.py --mode self_play --resume models/showdown_ppo_50000_steps
```

### Ladder (online play)

First, add your Showdown credentials to `config/config.yaml`:

```yaml
credentials:
  username: "YourUsername"
  password: "YourPassword"
```

Then run:

```bash
# Play and keep learning
python main.py --mode ladder --model models/final_model --battles 100

# Evaluate only (no training)
python main.py --mode ladder --model models/final_model --battles 50 --no-train
```

### Monitor training with TensorBoard

```bash
tensorboard --logdir logs/
```

---

## How It Works

### State Encoding (`src/state/encoder.py`)

Every battle state is converted into a fixed-size numeric vector containing:

| Component | Size | Description |
|-----------|------|-------------|
| Own active Pokemon | 138 | HP, types, status, boosts, stats, 4 moves |
| Own reserve (x5) | 100 | HP, types, availability |
| Opponent active Pokemon | 138 | HP, types, status, boosts (moves unknown) |
| Opponent reserve (x5) | 100 | HP, types (partial info) |
| Field conditions | 20 | Weather, terrain, screens |
| **Total** | **496** | |

### Action Space (`src/bot/action_space.py`)

9 discrete actions:
- `0-3`: Use move 1-4
- `4-8`: Switch to reserve Pokemon 1-5

Unavailable actions are masked so the agent never selects illegal moves.

### Reward Function (`src/agent/reward.py`)

| Event | Reward |
|-------|--------|
| Win | +1.0 |
| Lose | -1.0 |
| Opponent Pokemon faints | +0.15 |
| Own Pokemon faints | -0.15 |
| HP difference (end of battle) | ±0.01 |

All values are configurable in `config/config.yaml`.

### RL Algorithm

PPO (Proximal Policy Optimization) with a 2-layer MLP (256 units each). Hyperparameters are fully configurable in `config/config.yaml`.

---

## Configuration

Edit `config/config.yaml` to adjust:

- Server host/port
- Showdown credentials
- Battle format (default: `gen9randombattle`)
- Training timesteps and parallelism
- PPO hyperparameters
- Reward shaping coefficients

---

## Roadmap

- [x] Random Battle (Gen 9) support
- [x] Self-play training
- [x] Ladder integration
- [ ] OU (Gen 9) support with team building
- [ ] Action masking with sb3-contrib
- [ ] Opponent modeling
- [ ] Pre-trained model release

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## License

MIT
