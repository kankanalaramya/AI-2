# BEGIN AI ACKNOWLEDGMENT
# AI-assisted project preparation: OpenAI ChatGPT / Codex.
# Scope and disclosure: report/REPORT.md, Section 10.
# Detailed assistance record: report/AI_ASSISTANCE.md (Appendix C).
# Project repository: https://github.com/kankanalaramya/AI-2
# Conversation links omitted for privacy; see the disclosure in Section 10.
# END AI ACKNOWLEDGMENT
"""
Q-Learning for COSC-4117EL Assignment 2.
The agent interacts with the supplied environment and does not use its transition model.

AI assistance acknowledgment: see the source comments above, report Section 10, and Appendix C for scope and disclosure.
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from warehouse_gridworld_domain_random import WarehouseGridWorld, ACTIONS, SHELF
from value_iteration import value_iteration, terminal_state, draw_policy

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
                     gamma=GAMMA, epsilon_decay=EPSILON_DECAY, progress_callback=None):
    # Reset motion separately from the layout, so earlier evaluations cannot
    # change a training run with the same seed.
    world.np_random = np.random.default_rng(seed)
    rng = np.random.default_rng(seed + 10000)
    states = all_states(world)
    q = {s: np.zeros(len(ACTIONS), dtype=float) for s in states}
    rewards, successes, epsilons = [], [], []
    epsilon = EPSILON_START

    for episode in range(episodes):
        state = world.reset()
        total_reward = 0.0
        success = False

        for _ in range(world.max_steps):
            action_index = choose_action(q, state, epsilon, rng)
            action = ACTIONS[action_index]
            result = world.step(action, intended_probability=p)
            next_state = result.state
            total_reward += result.reward

            if result.done and result.info.get("event") == "return_to_dock":
                success = True

            # Only returning to the dock ends the underlying task. A timeout
            # stops this episode, but keeps the estimated value of its next state.
            if success:
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
        if progress_callback is not None:
            if (episode + 1) % 100 == 0 or episode + 1 == episodes:
                progress_callback(episode + 1, episodes,
                                  float(np.mean(rewards[-100:])),
                                  float(np.mean(successes[-100:])))

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
        for _ in range(world.max_steps):
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
        for _ in range(world.max_steps):
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
    episodes = np.arange(1, len(rewards) + 1)
    axes[0].plot(episodes, rewards, alpha=0.35, label="Episode reward")
    window = min(100, len(rewards))
    if window:
        moving = np.convolve(rewards, np.ones(window) / window, mode="valid")
        # Both averages belong at the LAST episode in their window.
        average_episodes = episodes[window - 1:]
        axes[0].plot(average_episodes, moving, label=f"{window}-episode moving average")
        success_average = np.convolve(successes, np.ones(window) / window, mode="valid")
        axes[1].plot(average_episodes, success_average)
    axes[0].set_ylabel("Cumulative reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel(f"{window}-episode success rate")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def draw_learned_policy(world, q, filename, title, phase=(1, 0)):
    # Use the same renderer and phase labels for both algorithms.
    policy = {}
    for state in q:
        if terminal_state(state, world):
            policy[state] = None
        else:
            policy[state] = ACTIONS[int(np.argmax(q[state]))]
    draw_policy(world, policy, filename, title, phase)


def main():
    os.makedirs("outputs", exist_ok=True)
    results = []
    # Required two hyperparameter configurations, multiple seeds, and all p settings.
    for config in CONFIGS:
        for p in PROBABILITIES:
            # Match the discount factor as well as the layout and motion model.
            world = WarehouseGridWorld(seed=42, max_steps=MAX_STEPS)
            vi_values, vi_policy, vi_iterations = value_iteration(world, p, gamma=config["gamma"])
            for phase, suffix in [((0, 0), "before_pickup"), ((1, 0), "carrying"), ((1, 1), "after_delivery")]:
                draw_policy(world, vi_policy,
                            f"outputs/{config['name']}_vi_p_{p:.2f}_{suffix}.png",
                            f"Value Iteration | p={p:.2f} | gamma={config['gamma']}", phase)
            for seed in SEEDS:
                # Keep the same warehouse layout (seed 42) for fair comparison with Value Iteration.
                # The experiment seed changes the agent exploration and motion random streams.
                world = WarehouseGridWorld(seed=42, max_steps=MAX_STEPS)
                evaluation_seed = seed + 500
                vi_eval_reward, vi_eval_success = evaluate_fixed_policy(
                    world, vi_policy, p, evaluation_seed, runs=100
                )
                q, rewards, successes, eps = train_q_learning(
                    world, p, seed, alpha=config["alpha"], gamma=config["gamma"],
                    epsilon_decay=config["epsilon_decay"]
                )
                prefix = f"outputs/{config['name']}_p_{p:.2f}_seed_{seed}"
                plot_learning(rewards, successes, prefix + "_learning.png",
                              f"Q-Learning {config['name']} | p={p:.2f} | seed={seed}")
                avg_reward, success_rate, path = evaluate_greedy(world, q, p, evaluation_seed)
                # Keep the measurements and one greedy run for the final report.
                with open(prefix + "_episodes.csv", "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["episode", "reward", "success", "epsilon"])
                    for episode in range(len(rewards)):
                        writer.writerow([episode + 1, rewards[episode], successes[episode], eps[episode]])
                with open(prefix + "_greedy_path.csv", "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["step", "row", "column", "has_package", "delivered"])
                    for step, state in enumerate(path):
                        writer.writerow([step] + list(state))
                for phase, suffix in [((0, 0), "before_pickup"), ((1, 0), "carrying"), ((1, 1), "after_delivery")]:
                    draw_learned_policy(world, q, prefix + "_" + suffix + ".png",
                                        f"Q-Learning {config['name']} | p={p:.2f} | seed={seed}", phase)
                results.append({
                    "configuration": config["name"], "p": p, "seed": seed,
                    "layout_seed": 42, "evaluation_seed": evaluation_seed, "evaluation_runs": 100,
                    "alpha": config["alpha"], "gamma": config["gamma"],
                    "epsilon_decay": config["epsilon_decay"],
                    "training_mean_last_100": float(np.mean(rewards[-100:])),
                    "evaluation_mean_reward": avg_reward,
                    "evaluation_success_rate": success_rate,
                    "value_iteration_iterations": vi_iterations,
                    "value_iteration_gamma": config["gamma"],
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

    # Compare both methods. Error bars show variation across training/evaluation seeds.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for config in CONFIGS:
        for method, prefix in [("Q-Learning", ""), ("Value Iteration", "value_iteration_")]:
            for ax, metric in zip(axes, ["evaluation_mean_reward", "evaluation_success_rate"]):
                means, deviations = [], []
                for p in PROBABILITIES:
                    vals = [row[prefix + metric] for row in results
                            if row["configuration"] == config["name"] and row["p"] == p]
                    means.append(np.mean(vals))
                    deviations.append(np.std(vals))
                style = "-" if method == "Q-Learning" else "--"
                ax.errorbar(PROBABILITIES, means, yerr=deviations, marker="o",
                            linestyle=style, capsize=4, label=f"{config['name']} {method}")
    for ax in axes:
        ax.set_xlabel("Intended-action probability")
        ax.set_xticks(PROBABILITIES)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Mean evaluation reward")
    axes[1].set_ylabel("Task success rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=8)
    fig.suptitle("Matched VI / Q-Learning comparisons | error bars: standard deviation across seeds")
    fig.tight_layout()
    fig.savefig("outputs/algorithm_probability_comparison.png", dpi=160)
    plt.close(fig)
    print("Q-Learning experiments complete. Check outputs/.")


if __name__ == "__main__":
    main()
