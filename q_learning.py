"""
Q-Learning for COSC-4117EL Assignment 2.
The agent interacts with the supplied environment and does not use its transition model.

AI assistance acknowledgment: this starter implementation was prepared with ChatGPT assistance. Add the actual conversation/share link in your submitted code/report as required by the assignment.
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from warehouse_gridworld_domain_random import WarehouseGridWorld, ACTIONS, SHELF
from value_iteration import value_iteration, terminal_state, ARROWS

EPISODES = 2500
MAX_STEPS = 120
GAMMA = 0.95
ALPHA = 0.15
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
SEEDS = [11, 22, 33]
PROBABILITIES = (1.00, 0.90, 0.80)
CONFIGS = [
    {"name": "config_A", "alpha": 0.15, "gamma": 0.95, "epsilon_decay": 0.995},
    {"name": "config_B", "alpha": 0.30, "gamma": 0.90, "epsilon_decay": 0.990},
]


def all_states(world):
    states = []
    for r in range(world.size):
        for c in range(world.size):
            if world.grid[r, c] != SHELF:
                states.extend([(r, c, 0, 0), (r, c, 1, 0), (r, c, 1, 1)])
    return states


def choose_action(q, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(ACTIONS)))
    return int(np.argmax(q[state]))


def train_q_learning(world, p, seed, episodes=EPISODES, alpha=ALPHA,
                     gamma=GAMMA, epsilon_decay=EPSILON_DECAY):
    rng = np.random.default_rng(seed + 10000)
    states = all_states(world)
    q = {s: np.zeros(len(ACTIONS), dtype=float) for s in states}
    rewards, successes, epsilons = [], [], []
    epsilon = EPSILON_START

    for episode in range(episodes):
        state = world.reset()
        total_reward = 0.0
        success = False

        for _ in range(MAX_STEPS):
            action_index = choose_action(q, state, epsilon, rng)
            action = ACTIONS[action_index]
            result = world.step(action, intended_probability=p)
            next_state = result.state
            total_reward += result.reward

            if result.done and result.info.get("event") == "return_to_dock":
                success = True

            if result.done:
                target = result.reward
            else:
                target = result.reward + gamma * float(np.max(q[next_state]))

            q[state][action_index] += alpha * (target - q[state][action_index])
            state = next_state
            if result.done:
                break

        rewards.append(total_reward)
        successes.append(int(success))
        epsilons.append(epsilon)
        epsilon = max(EPSILON_MIN, epsilon * epsilon_decay)

    return q, np.asarray(rewards), np.asarray(successes), np.asarray(epsilons)


def evaluate_greedy(world, q, p, seed, runs=100):
    rng = np.random.default_rng(seed)
    episode_rewards, episode_success = [], []
    trajectories = []
    for run in range(runs):
        # Reseed only the motion RNG, retaining the same fixed layout.
        world.np_random = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        state = world.reset()
        total = 0.0
        path = [state]
        success = False
        for _ in range(MAX_STEPS):
            action_index = int(np.argmax(q[state]))
            result = world.step(ACTIONS[action_index], intended_probability=p)
            state = result.state
            path.append(state)
            total += result.reward
            if result.done:
                success = result.info.get("event") == "return_to_dock"
                break
        episode_rewards.append(total)
        episode_success.append(int(success))
        if run == 0:
            trajectories = path
    return float(np.mean(episode_rewards)), float(np.mean(episode_success)), trajectories



def evaluate_fixed_policy(world, policy, p, seed, runs=100):
    """Evaluate a deterministic Value Iteration policy through environment interaction."""
    rng = np.random.default_rng(seed)
    episode_rewards, episode_success = [], []
    for _ in range(runs):
        world.np_random = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        state = world.reset()
        total = 0.0
        success = False
        for _ in range(MAX_STEPS):
            action = policy.get(state)
            if action is None:
                break
            result = world.step(action, intended_probability=p)
            state = result.state
            total += result.reward
            if result.done:
                success = result.info.get("event") == "return_to_dock"
                break
        episode_rewards.append(total)
        episode_success.append(int(success))
    return float(np.mean(episode_rewards)), float(np.mean(episode_success))

def plot_learning(rewards, successes, filename, title):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(rewards, alpha=0.35, label="Episode reward")
    window = min(100, len(rewards))
    if window:
        moving = np.convolve(rewards, np.ones(window) / window, mode="valid")
        axes[0].plot(np.arange(window - 1, len(rewards)), moving, label=f"{window}-episode moving average")
    axes[0].set_ylabel("Cumulative reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(np.convolve(successes, np.ones(100) / 100, mode="valid"))
    axes[1].set_ylabel("100-episode success rate")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def draw_learned_policy(world, q, filename, title, phase=(1, 0)):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(world.grid, cmap="Pastel1", vmin=0, vmax=5)
    for r in range(world.size):
        for c in range(world.size):
            cell = int(world.grid[r, c])
            label = "#" if cell == 1 else "!" if cell == 2 else ""
            if (r, c) == world.pickup_pos: label = "P"
            if (r, c) == world.packing_pos: label = "D"
            if (r, c) == world.dock_pos: label = "C"
            ax.text(c, r, label, ha="center", va="center", fontsize=10, fontweight="bold")
            has_package, delivered = phase
            s = (r, c, has_package, delivered)
            if s in q and cell != SHELF:
                ax.text(c, r + 0.28, ARROWS[ACTIONS[int(np.argmax(q[s]))]],
                        ha="center", va="center", fontsize=10, color="navy")
    ax.set_title(title + "\nArrows show learned policy while carrying package")
    ax.set_xticks(range(world.size)); ax.set_yticks(range(world.size))
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def main():
    os.makedirs("outputs", exist_ok=True)
    results = []
    # Required two hyperparameter configurations, multiple seeds, and all p settings.
    for config in CONFIGS:
        for p in PROBABILITIES:
            for seed in SEEDS:
                # Keep the same warehouse layout (seed 42) for fair comparison with Value Iteration.
                # The experiment seed changes the agent exploration and motion random streams.
                world = WarehouseGridWorld(seed=42, max_steps=MAX_STEPS)
                world.np_random = np.random.default_rng(seed)
                vi_values, vi_policy, vi_iterations = value_iteration(world, p)
                vi_eval_reward, vi_eval_success = evaluate_fixed_policy(
                    world, vi_policy, p, seed + 900, runs=100
                )
                q, rewards, successes, eps = train_q_learning(
                    world, p, seed, alpha=config["alpha"], gamma=config["gamma"],
                    epsilon_decay=config["epsilon_decay"]
                )
                prefix = f"outputs/{config['name']}_p_{p:.2f}_seed_{seed}"
                plot_learning(rewards, successes, prefix + "_learning.png",
                              f"Q-Learning {config['name']} | p={p:.2f} | seed={seed}")
                avg_reward, success_rate, path = evaluate_greedy(world, q, p, seed + 500)
                for phase, suffix in [((0, 0), "before_pickup"), ((1, 0), "carrying"), ((1, 1), "after_delivery")]:
                    draw_learned_policy(world, q, prefix + "_" + suffix + ".png",
                                        f"Q-Learning {config['name']} | p={p:.2f} | seed={seed}", phase)
                results.append({
                    "configuration": config["name"], "p": p, "seed": seed,
                    "alpha": config["alpha"], "gamma": config["gamma"],
                    "epsilon_decay": config["epsilon_decay"],
                    "training_mean_last_100": float(np.mean(rewards[-100:])),
                    "evaluation_mean_reward": avg_reward,
                    "evaluation_success_rate": success_rate,
                    "value_iteration_iterations": vi_iterations,
                    "value_iteration_evaluation_mean_reward": vi_eval_reward,
                    "value_iteration_evaluation_success_rate": vi_eval_success,
                    "episodes": EPISODES
                })
                print(f"{config['name']} p={p:.2f} seed={seed}: "
                      f"eval reward={avg_reward:.2f}, success={success_rate:.1%}")

    with open("outputs/q_learning_experiment_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # One concise comparison plot: mean evaluation reward by p and configuration.
    fig, ax = plt.subplots(figsize=(8, 5))
    for config in CONFIGS:
        means = []
        for p in PROBABILITIES:
            vals = [x["evaluation_mean_reward"] for x in results
                    if x["configuration"] == config["name"] and x["p"] == p]
            means.append(np.mean(vals))
        ax.plot(PROBABILITIES, means, marker="o", label=config["name"])
    ax.set_xlabel("Intended-action probability")
    ax.set_ylabel("Mean greedy evaluation reward")
    ax.set_title("Q-Learning performance under stochastic motion")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/q_learning_probability_comparison.png", dpi=160)
    plt.close(fig)
    print("Q-Learning experiments complete. Check outputs/.")


if __name__ == "__main__":
    main()
