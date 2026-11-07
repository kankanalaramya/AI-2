"""
Warehouse Robot GUI for COSC-4117EL Assignment 2.

Run from the project folder:
    python warehouse_robot_gui.py

Controls
--------
M       Manual control mode
V       Calculate and run the Value Iteration policy
Q       Train and run the Q-Learning policy
SPACE   Play/pause automatic AI movement
Arrow keys  Move manually (or single-step while AI is paused)
1 / 2 / 3  Set intended-action probability to 1.00 / 0.90 / 0.80
R       Reset current episode
N       Generate the next seeded warehouse layout
+ / -   Increase / decrease AI playback speed
ESC     Quit

AI assistance acknowledgment: this GUI was created with ChatGPT assistance.
Add the required conversation/share link to the submitted report/code.
"""

import sys
import pygame
import numpy as np

from warehouse_gridworld_domain_random import (
    WarehouseGridWorld, ACTIONS, SHELF, CONGESTION, PICKUP, PACKING, DOCK,
)
from value_iteration import value_iteration
from q_learning import train_q_learning

# --------------------------- Display settings ---------------------------
CELL = 62
GRID_PIXELS = CELL * 10
PANEL_WIDTH = 340
WINDOW_WIDTH = GRID_PIXELS + PANEL_WIDTH
WINDOW_HEIGHT = GRID_PIXELS + 90
FPS = 60

WHITE = (250, 251, 253)
FLOOR = (242, 245, 249)
SHELF_COLOR = (45, 49, 57)
CONGESTION_COLOR = (244, 174, 174)
PICKUP_COLOR = (112, 170, 255)
PACKING_COLOR = (255, 220, 91)
DOCK_COLOR = (113, 210, 155)
ROBOT_COLOR = (116, 61, 184)
GRID_COLOR = (190, 198, 208)
TEXT = (35, 42, 52)
MUTED = (100, 110, 122)
ACCENT = (45, 105, 175)
GREEN = (34, 139, 85)
ORANGE = (205, 125, 35)

ARROWS = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
PROBABILITIES = {pygame.K_1: 1.00, pygame.K_2: 0.90, pygame.K_3: 0.80}


class WarehouseRobotApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Warehouse Robot — Manual Control & AI Policies")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 20, bold=True)
        self.small = pygame.font.SysFont("arial", 16)
        self.tiny = pygame.font.SysFont("arial", 13)
        self.title_font = pygame.font.SysFont("arial", 24, bold=True)

        self.world = WarehouseGridWorld(seed=42, max_steps=120)
        self.mode = "MANUAL"
        self.probability = 1.0
        self.policy = None
        self.q_table = None
        self.playing = False
        self.delay_ms = 450
        self.last_move_time = 0
        self.message = "Manual mode: use the arrow keys to move."
        self.running = True

    def invalidate_ai(self):
        self.policy = None
        self.q_table = None
        self.playing = False

    def set_probability(self, p):
        if p != self.probability:
            self.probability = p
            self.invalidate_ai()
            if self.mode != "MANUAL":
                self.mode = "MANUAL"
            self.message = (
                f"Movement probability set to {p:.2f}. "
                "Select V or Q to calculate a new policy."
            )

    def calculate_value_iteration(self):
        self.playing = False
        self.message = "Calculating Value Iteration policy..."
        self.draw()
        pygame.display.flip()
        values, policy, iterations = value_iteration(
            self.world, intended_probability=self.probability
        )
        self.policy = policy
        self.q_table = None
        self.mode = "VALUE ITERATION"
        self.world.reset()
        self.world.np_random = np.random.default_rng(4200)
        self.playing = True
        self.message = (
            f"Value Iteration ready: {iterations} iterations, "
            f"V(start)={values[self.world.get_state()]:.2f}."
        )

    def train_q_policy(self):
        self.playing = False
        self.message = "Training Q-Learning (2,500 episodes)... Please wait."
        self.draw()
        pygame.display.flip()
        self.world.reset()
        self.world.np_random = np.random.default_rng(42)
        q, rewards, successes, epsilons = train_q_learning(
            self.world,
            self.probability,
            seed=42,
            episodes=2500,
            alpha=0.15,
            gamma=0.95,
            epsilon_decay=0.995,
        )
        self.q_table = q
        self.policy = None
        self.mode = "Q-LEARNING"
        self.world.reset()
        self.world.np_random = np.random.default_rng(4200)
        self.playing = True
        final_reward = float(np.mean(rewards[-100:]))
        final_success = float(np.mean(successes[-100:]))
        self.message = (
            f"Q-Learning ready. Last 100 training episodes: "
            f"reward {final_reward:.1f}, success {final_success:.0%}."
        )

    def current_ai_action(self):
        state = self.world.get_state()
        if self.mode == "VALUE ITERATION" and self.policy is not None:
            return self.policy.get(state)
        if self.mode == "Q-LEARNING" and self.q_table is not None:
            if state in self.q_table:
                return ACTIONS[int(np.argmax(self.q_table[state]))]
        return None

    def step_robot(self, action):
        result = self.world.step(action, intended_probability=self.probability)
        event = result.info.get("event", "move")
        actual = result.info.get("actual_action", action)
        if actual != action:
            self.message = (
                f"Intended {action}; actual movement {actual} "
                f"(stochastic slip). Reward: {result.reward:+.0f}."
            )
        else:
            self.message = f"{event.replace('_', ' ').title()}: reward {result.reward:+.0f}."
        if result.done:
            self.playing = False
            if result.info.get("event") == "return_to_dock":
                self.message = "SUCCESS! Package delivered and robot returned to the dock."
            else:
                self.message = "Episode ended: maximum step limit reached."

    def reset_episode(self):
        self.world.reset()
        # Keep AI policy but reset the motion stream for repeatable demonstrations.
        self.world.np_random = np.random.default_rng(4200)
        self.playing = False
        self.message = "Episode reset. Press SPACE to resume AI or use arrow keys."

    def new_layout(self):
        next_seed = self.world.seed_value + 1
        self.world.regenerate_layout(next_seed)
        self.invalidate_ai()
        self.mode = "MANUAL"
        self.message = (
            f"New warehouse layout generated (seed {next_seed}). "
            "AI policies need to be recalculated."
        )

    def handle_key(self, key):
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_m:
            self.mode = "MANUAL"
            self.playing = False
            self.message = "Manual mode: use the arrow keys to move."
        elif key == pygame.K_v:
            try:
                self.calculate_value_iteration()
            except Exception as exc:
                self.playing = False
                self.message = f"Value Iteration error: {exc}"
        elif key == pygame.K_q:
            try:
                self.train_q_policy()
            except Exception as exc:
                self.playing = False
                self.message = f"Q-Learning error: {exc}"
        elif key in PROBABILITIES:
            self.set_probability(PROBABILITIES[key])
        elif key == pygame.K_r:
            self.reset_episode()
        elif key == pygame.K_n:
            self.new_layout()
        elif key == pygame.K_SPACE:
            if self.mode == "MANUAL":
                self.message = "Choose V (Value Iteration) or Q (Q-Learning) for automatic play."
            elif self.policy is None and self.q_table is None:
                self.message = "No AI policy available. Press V or Q first."
            elif self.world.done:
                self.message = "Episode finished. Press R to reset."
            else:
                self.playing = not self.playing
                self.message = "AI playback running." if self.playing else "AI playback paused."
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.delay_ms = max(100, self.delay_ms - 75)
            self.message = f"Playback speed increased ({self.delay_ms} ms per move)."
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.delay_ms = min(1200, self.delay_ms + 75)
            self.message = f"Playback speed decreased ({self.delay_ms} ms per move)."
        elif key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
            action = {
                pygame.K_UP: "up",
                pygame.K_DOWN: "down",
                pygame.K_LEFT: "left",
                pygame.K_RIGHT: "right",
            }[key]
            if not self.world.done and (self.mode == "MANUAL" or not self.playing):
                self.step_robot(action)

    def draw_text(self, text, x, y, color=TEXT, font=None):
        surf = (font or self.small).render(str(text), True, color)
        self.screen.blit(surf, (x, y))

    def draw_grid(self):
        for r in range(self.world.size):
            for c in range(self.world.size):
                cell = int(self.world.grid[r, c])
                color = FLOOR
                if cell == SHELF:
                    color = SHELF_COLOR
                elif cell == CONGESTION:
                    color = CONGESTION_COLOR
                elif cell == PICKUP:
                    color = PICKUP_COLOR
                elif cell == PACKING:
                    color = PACKING_COLOR
                elif cell == DOCK:
                    color = DOCK_COLOR

                rect = pygame.Rect(c * CELL, r * CELL, CELL, CELL)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, GRID_COLOR, rect, 1)

                labels = {
                    SHELF: "#",
                    CONGESTION: "!",
                    PICKUP: "P",
                    PACKING: "D",
                    DOCK: "C",
                }
                if cell in labels:
                    label_color = WHITE if cell == SHELF else TEXT
                    surf = self.font.render(labels[cell], True, label_color)
                    self.screen.blit(surf, surf.get_rect(center=rect.center))

                # Show policy arrows only when an AI policy is loaded.
                if self.mode == "VALUE ITERATION" and self.policy is not None:
                    action = self.policy.get((r, c, int(self.world.has_package), int(self.world.delivered)))
                    if action:
                        arrow = self.font.render(ARROWS[action], True, ACCENT)
                        self.screen.blit(arrow, (c * CELL + 42, r * CELL + 34))
                elif self.mode == "Q-LEARNING" and self.q_table is not None:
                    state = (r, c, int(self.world.has_package), int(self.world.delivered))
                    if state in self.q_table and cell != SHELF:
                        action = ACTIONS[int(np.argmax(self.q_table[state]))]
                        arrow = self.font.render(ARROWS[action], True, ACCENT)
                        self.screen.blit(arrow, (c * CELL + 42, r * CELL + 34))

        # Robot marker
        cx = int((self.world.robot_pos[1] + 0.5) * CELL)
        cy = int((self.world.robot_pos[0] + 0.5) * CELL)
        pygame.draw.circle(self.screen, ROBOT_COLOR, (cx, cy), CELL // 3)
        pygame.draw.circle(self.screen, WHITE, (cx, cy), CELL // 3, 2)
        robot_label = self.font.render("R", True, WHITE)
        self.screen.blit(robot_label, robot_label.get_rect(center=(cx, cy)))

    def draw_panel(self):
        x = GRID_PIXELS + 22
        self.draw_text("WAREHOUSE ROBOT", x, 20, ACCENT, self.title_font)
        self.draw_text("Assignment 2 • Grid World", x, 55, MUTED)

        y = 85
        self.draw_text("CURRENT MODE", x, y, MUTED, self.tiny)
        self.draw_text(self.mode, x, y + 20, GREEN if self.mode != "MANUAL" else ACCENT, self.font)

        y += 65
        self.draw_text("TASK PROGRESS", x, y, MUTED, self.tiny)
        stages = [
            ("1. Pick up package", self.world.has_package),
            ("2. Deliver to packing", self.world.delivered),
            ("3. Return to charging dock", self.world.done and self.world.delivered),
        ]
        for i, (label, complete) in enumerate(stages):
            self.draw_text(("✓ " if complete else "○ ") + label, x, y + 22 + i * 24,
                           GREEN if complete else TEXT, self.small)

        y += 100
        self.draw_text("LIVE STATISTICS", x, y, MUTED, self.tiny)
        stats = [
            f"Position: {self.world.robot_pos}",
            f"Score: {self.world.score:.0f}",
            f"Steps: {self.world.steps}/{self.world.max_steps}",
            f"Movement p: {self.probability:.2f}",
            f"Playback: {self.delay_ms} ms / move",
        ]
        for i, line in enumerate(stats):
            self.draw_text(line, x, y + 22 + i * 22, TEXT, self.small)

        y += 120
        self.draw_text("WAREHOUSE LEGEND", x, y, MUTED, self.tiny)
        legend = [
            ("C", "Charging dock", DOCK_COLOR),
            ("P", "Pickup station", PICKUP_COLOR),
            ("D", "Packing station", PACKING_COLOR),
            ("#", "Shelf / blocked", SHELF_COLOR),
            ("!", "Congestion / penalty", CONGESTION_COLOR),
            ("R", "Robot", ROBOT_COLOR),
        ]
        for i, (symbol, label, color) in enumerate(legend):
            yy = y + 20 + i * 23
            pygame.draw.rect(self.screen, color, (x, yy, 20, 18), border_radius=3)
            fg = WHITE if symbol == "#" else TEXT
            s = self.tiny.render(symbol, True, fg)
            self.screen.blit(s, s.get_rect(center=(x + 10, yy + 9)))
            self.draw_text(label, x + 30, yy + 1, TEXT, self.tiny)

        y += 150
        self.draw_text("CONTROLS", x, y, MUTED, self.tiny)
        controls = [
            "M: Manual   V: Value Iteration",
            "Q: Train Q-Learning",
            "SPACE: Play / pause AI",
            "Arrows: Move / single step",
            "1 / 2 / 3: Set p = 1.0 / 0.9 / 0.8",
            "R: Reset   N: New layout",
            "+ / -: Playback speed   ESC: Quit",
        ]
        for i, line in enumerate(controls):
            self.draw_text(line, x, y + 19 + i * 19, TEXT, self.tiny)

    def draw(self):
        self.screen.fill(WHITE)
        self.draw_grid()
        pygame.draw.rect(self.screen, (247, 249, 252),
                         (0, GRID_PIXELS, WINDOW_WIDTH, 90))
        pygame.draw.line(self.screen, GRID_COLOR, (0, GRID_PIXELS),
                         (WINDOW_WIDTH, GRID_PIXELS), 2)
        self.draw_text(
            f"Seed {self.world.seed_value}  |  Pickup {self.world.pickup_pos}  "
            f"Packing {self.world.packing_pos}  Dock {self.world.dock_pos}",
            12, GRID_PIXELS + 10, TEXT, self.small
        )
        message = self.message
        if len(message) > 112:
            message = message[:109] + "..."
        self.draw_text(message, 12, GRID_PIXELS + 42, ACCENT, self.tiny)
        self.draw_panel()

    def run(self):
        while self.running:
            now = pygame.time.get_ticks()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)

            if self.playing and now - self.last_move_time >= self.delay_ms:
                action = self.current_ai_action()
                if action is None:
                    self.playing = False
                    self.message = "AI policy has no action for this state. Press R to reset."
                elif not self.world.done:
                    self.step_robot(action)
                self.last_move_time = now

            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        return 0


if __name__ == "__main__":
    try:
        app = WarehouseRobotApp()
        sys.exit(app.run())
    except Exception as error:
        pygame.quit()
        print(f"Could not start Warehouse Robot GUI: {error}")
        raise
