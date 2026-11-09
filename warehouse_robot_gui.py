# BEGIN AI ACKNOWLEDGMENT
# AI-assisted project preparation: OpenAI ChatGPT / Codex.
# Scope and disclosure: report/REPORT.md, Section 10.
# Detailed assistance record: report/AI_ASSISTANCE.md (Appendix C).
# Project repository: https://github.com/kankanalaramya/AI-2
# Conversation links omitted for privacy; see the disclosure in Section 10.
# END AI ACKNOWLEDGMENT
"""
Warehouse Robot GUI for COSC-4117EL Assignment 2.
Run: python warehouse_robot_gui.py

M: manual, V: Value Iteration, Q: Q-Learning, Space: play/pause,
S: one policy step, H: policy arrows, R: reset, N: new layout,
1/2/3: motion probability, arrow keys: manual move, +/-: speed, Esc: quit.

AI assistance acknowledgment: ChatGPT assisted with this GUI and its cleanup.
See the source comments above, report Section 10, and Appendix C for scope and disclosure.
"""

import pygame
import numpy as np

from warehouse_gridworld_domain_random import (
    WarehouseGridWorld, ACTIONS, SHELF, CONGESTION, PICKUP, PACKING, DOCK,
)
from value_iteration import value_iteration, terminal_state, ARROWS
from q_learning import train_q_learning

WINDOW_WIDTH = 1080
WINDOW_HEIGHT = 820
CELL = 54
GRID_X = 44
GRID_Y = 180
FPS = 60

BACKGROUND = (241, 244, 248)
WHITE = (255, 255, 255)
TEXT = (32, 43, 58)
MUTED = (94, 108, 126)
ACCENT = (37, 89, 157)
BORDER = (211, 220, 230)
GREEN = (30, 116, 77)
ROBOT_COLOR = (111, 58, 162)
CELL_COLORS = [
    (250, 251, 253), (48, 56, 68), (242, 177, 168),
    (137, 181, 239), (242, 210, 112), (145, 212, 177),
]
PROBABILITIES = {pygame.K_1: 1.00, pygame.K_2: 0.90, pygame.K_3: 0.80}


class WarehouseRobotApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Warehouse Robot")
        self.window = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE
        )
        # Keep simple drawing coordinates; scale the finished interface on resize.
        self.screen = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT)).convert()
        self.clock = pygame.time.Clock()
        self.small = pygame.font.SysFont("dejavusans", 16)
        self.tiny = pygame.font.SysFont("dejavusans", 13)
        self.font = pygame.font.SysFont("dejavusans", 20, bold=True)
        self.title_font = pygame.font.SysFont("dejavusans", 28, bold=True)

        self.world = WarehouseGridWorld(seed=42, max_steps=120)
        self.probability = 0.80
        self.mode = "Manual"
        self.policy = None
        self.q_table = None
        self.show_policy = False
        self.playing = False
        self.training = False
        self.delay_ms = 450
        self.last_move_time = 0
        self.running = True
        self.buttons = []
        self.algorithm_lines = ["Choose an algorithm above the grid.", "Manual control uses the arrow keys."]
        self.reset_episode()
        self.message = "Visit pickup, packing, then the dock. Choose a mode to begin."

    def task_complete(self):
        return (self.world.done and self.world.delivered
                and self.world.robot_pos == self.world.dock_pos)

    def invalidate_ai(self):
        self.policy = None
        self.q_table = None
        self.playing = False
        self.mode = "Manual"
        self.algorithm_lines = ["Choose an algorithm above the grid.", "Manual control uses the arrow keys."]

    def reset_episode(self):
        self.world.reset()
        self.world.np_random = np.random.default_rng(4200)
        self.playing = False
        self.last_move = "No move yet."
        self.message = "Episode reset. Play the policy or move with the arrow keys."

    def set_probability(self, p):
        if p != self.probability:
            self.probability = p
            self.invalidate_ai()
            self.reset_episode()
            self.message = f"Motion probability is now {p:.2f}. Calculate or train a new policy."

    def new_layout(self):
        self.world.regenerate_layout(self.world.seed_value + 1)
        self.invalidate_ai()
        self.reset_episode()
        self.message = "New station locations. Calculate or train a policy for this layout."

    def calculate_value_iteration(self):
        self.invalidate_ai()
        self.reset_episode()
        self.message = "Calculating Value Iteration..."
        self.draw()
        pygame.display.flip()
        values, policy, iterations = value_iteration(self.world, self.probability)
        self.policy = policy
        self.mode = "Value Iteration"
        self.algorithm_lines = [
            f"Converged in {iterations} iterations. Gamma: 0.95.",
            f"Expected discounted return: {values[self.world.get_state()]:.2f}.",
        ]
        self.message = "Policy ready. Press Play or Space to start; S takes one policy step."

    def training_progress(self, episode, episodes, reward, success):
        # Check events between training batches so the window can repaint/close.
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False
                raise InterruptedError("Training stopped.")
        self.algorithm_lines = [
            f"Training episode {episode:,} / {episodes:,}",
            f"Last 100: reward {reward:.1f}, success {success:.0%}.",
        ]
        self.draw()
        pygame.display.flip()

    def train_q_policy(self):
        self.invalidate_ai()
        self.reset_episode()
        self.mode = "Q-Learning"
        self.training = True
        self.algorithm_lines = ["Training episode 0 / 2,500", "Alpha: 0.15 | Gamma: 0.95 | Seed: 42"]
        self.message = "Learning from experience. Progress updates every 100 episodes."
        self.draw()
        pygame.display.flip()
        try:
            q, rewards, successes, epsilons = train_q_learning(
                self.world, self.probability, seed=42, episodes=2500,
                alpha=0.15, gamma=0.95, epsilon_decay=0.995,
                progress_callback=self.training_progress,
            )
        finally:
            self.training = False
        self.q_table = q
        self.policy = {}
        for state in q:
            if terminal_state(state, self.world):
                self.policy[state] = None
            else:
                self.policy[state] = ACTIONS[int(np.argmax(q[state]))]
        self.reset_episode()
        self.algorithm_lines = [
            "Trained for 2,500 episodes. Exploration off.",
            f"Last 100: reward {np.mean(rewards[-100:]):.1f}, success {np.mean(successes[-100:]):.0%}.",
        ]
        self.message = "Greedy policy ready. Press Play or Space; S takes one policy step."

    def current_ai_action(self):
        if self.policy is not None and not self.world.done:
            return self.policy.get(self.world.get_state())
        return None

    def step_robot(self, action):
        if self.world.done:
            return
        result = self.world.step(action, intended_probability=self.probability)
        actual = result.info["actual_action"]
        self.last_move = f"{action.title()} requested / {actual} taken | Reward {result.reward:+.0f}"
        event = result.info["event"]
        if event == "return_to_dock":
            self.message = "Task complete! Package delivered and robot returned to the dock."
        elif event == "max_steps":
            self.message = "Time limit reached. The task is incomplete. Reset to try again."
        elif event == "pickup":
            self.message = "Package collected. Next stop: the packing station."
        elif event == "delivery":
            self.message = "Package delivered. Next stop: the charging dock."
        elif event == "invalid_move":
            self.message = "Blocked by a shelf or boundary. The robot stayed in place."
        elif result.info.get("congestion"):
            self.message = "Congestion zone: an extra 10-point penalty."
        elif actual != action:
            self.message = "Sideways slip: the actual move differed from the requested move."
        else:
            self.message = "Moving toward the next station."
        if result.done:
            self.playing = False

    def handle_key(self, key):
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_m:
            self.invalidate_ai()
            self.message = "Manual control. Use the arrow keys to move."
        elif key in (pygame.K_v, pygame.K_q):
            try:
                if key == pygame.K_v:
                    self.calculate_value_iteration()
                else:
                    self.train_q_policy()
            except Exception as error:
                self.invalidate_ai()
                self.message = f"Could not prepare policy: {error}"
                print(self.message)
        elif key in PROBABILITIES:
            self.set_probability(PROBABILITIES[key])
        elif key == pygame.K_r:
            self.reset_episode()
        elif key == pygame.K_n:
            self.new_layout()
        elif key == pygame.K_h:
            self.show_policy = not self.show_policy
        elif key in (pygame.K_SPACE, pygame.K_s):
            action = self.current_ai_action()
            if action is None:
                self.message = "Reset a finished episode, or choose an algorithm to load a policy."
            elif key == pygame.K_s:
                self.playing = False
                self.step_robot(action)
            else:
                self.playing = not self.playing
                self.last_move_time = pygame.time.get_ticks()
                self.message = "Policy running." if self.playing else "Paused. Press S for one policy step."
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.delay_ms = max(100, self.delay_ms - 75)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.delay_ms = min(1200, self.delay_ms + 75)
        elif key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
            if not self.playing:
                actions = {pygame.K_UP: "up", pygame.K_DOWN: "down", pygame.K_LEFT: "left", pygame.K_RIGHT: "right"}
                self.step_robot(actions[key])

    def handle_click(self, position):
        # Convert the resized window's mouse coordinates back to drawing coordinates.
        viewport = self.screen.get_rect().fit(self.window.get_rect())
        if not viewport.collidepoint(position):
            return
        x = (position[0] - viewport.x) * WINDOW_WIDTH / viewport.width
        y = (position[1] - viewport.y) * WINDOW_HEIGHT / viewport.height
        for rect, key in self.buttons:
            if rect.collidepoint(x, y):
                self.handle_key(key)
                break

    def draw_text(self, text, x, y, color=TEXT, font=None):
        surface = (font or self.small).render(str(text), True, color)
        self.screen.blit(surface, (x, y))

    def draw_message(self, text, x, y, width):
        lines = []
        line = ""
        for word in text.split():
            candidate = (line + " " + word).strip()
            if line and self.small.size(candidate)[0] > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
        for index, line in enumerate(lines[:2]):
            if index == 1 and len(lines) > 2:
                line = line[:-3] + "..."
            self.draw_text(line, x, y + index * 22)

    def draw_card(self, y, height):
        rect = pygame.Rect(620, y, 436, height)
        pygame.draw.rect(self.screen, WHITE, rect, border_radius=10)
        pygame.draw.rect(self.screen, BORDER, rect, 1, border_radius=10)

    def draw_button(self, rect, label, key, selected=False, enabled=True):
        rect = pygame.Rect(rect)
        enabled = enabled and not self.training
        color = ACCENT if selected else WHITE
        pygame.draw.rect(self.screen, color, rect, border_radius=6)
        pygame.draw.rect(self.screen, BORDER, rect, 1, border_radius=6)
        text_color = WHITE if selected else TEXT
        if not enabled:
            text_color = MUTED
        label_surface = self.tiny.render(label, True, text_color)
        self.screen.blit(label_surface, label_surface.get_rect(center=rect.center))
        if enabled:
            self.buttons.append((rect, key))

    def draw_grid(self):
        for index in range(10):
            self.draw_text(index, GRID_X + index * CELL + 23, GRID_Y - 23, MUTED, self.tiny)
            self.draw_text(index, GRID_X - 23, GRID_Y + index * CELL + 18, MUTED, self.tiny)
        labels = {CONGESTION: "!", PICKUP: "P", PACKING: "D", DOCK: "C"}
        for row in range(10):
            for col in range(10):
                cell = int(self.world.grid[row, col])
                rect = pygame.Rect(GRID_X + col * CELL, GRID_Y + row * CELL, CELL, CELL)
                pygame.draw.rect(self.screen, CELL_COLORS[cell], rect)
                pygame.draw.rect(self.screen, BORDER, rect, 1)
                if cell in labels:
                    label = self.font.render(labels[cell], True, TEXT)
                    self.screen.blit(label, label.get_rect(center=rect.center))
                if self.show_policy and self.policy is not None and cell != SHELF:
                    state = (row, col, int(self.world.has_package), int(self.world.delivered))
                    action = self.policy.get(state)
                    if action:
                        self.draw_text(ARROWS[action], rect.right - 18, rect.bottom - 22, ACCENT)
        row, col = self.world.robot_pos
        center = (GRID_X + col * CELL + CELL // 2, GRID_Y + row * CELL + CELL // 2)
        pygame.draw.circle(self.screen, ROBOT_COLOR, center, 18)
        pygame.draw.circle(self.screen, WHITE, center, 18, 2)
        label = self.small.render("R", True, WHITE)
        self.screen.blit(label, label.get_rect(center=center))

    def draw_panel(self):
        complete = self.task_complete()
        if self.training:
            objective = "Training in progress"
        elif complete:
            objective = "Task complete"
        elif self.world.done:
            objective = "Time limit reached"
        elif not self.world.has_package:
            objective = "Collect the package"
        elif not self.world.delivered:
            objective = "Deliver to packing"
        else:
            objective = "Return to the dock"
        self.draw_card(88, 172)
        self.draw_text("NEXT STOP", 640, 103, MUTED, self.tiny)
        self.draw_text(objective, 640, 126, GREEN if complete else TEXT, self.font)
        stages = [
            ("Pickup", self.world.pickup_pos, self.world.has_package),
            ("Packing", self.world.packing_pos, self.world.delivered),
            ("Dock", self.world.dock_pos, complete),
        ]
        next_found = False
        for index, (name, position, done) in enumerate(stages):
            status = "Done" if done else "Pending"
            if not done and not next_found:
                if not self.world.done:
                    status = "Next"
                next_found = True
            y = 164 + index * 28
            self.draw_text(f"{index + 1}  {name}  {position}", 640, y)
            self.draw_text(status, 975, y, GREEN if done else MUTED, self.tiny)

        self.draw_card(274, 110)
        self.draw_text("TOTAL REWARD", 640, 289, MUTED, self.tiny)
        self.draw_text(f"{self.world.score:+.0f}", 640, 310, TEXT, self.title_font)
        self.draw_text("STEPS", 850, 289, MUTED, self.tiny)
        self.draw_text(f"{self.world.steps} / {self.world.max_steps}", 850, 310, TEXT, self.title_font)
        self.draw_text(f"Robot {self.world.robot_pos}   |   {self.delay_ms} ms / move (+/-)", 640, 355, MUTED, self.tiny)

        self.draw_card(398, 112)
        self.draw_text("MOVE RELIABILITY", 640, 412, MUTED, self.tiny)
        for index, (key, p) in enumerate(PROBABILITIES.items()):
            self.draw_button((640 + index * 134, 437, 126, 32), f"{p:.0%}  [{index + 1}]", key, p == self.probability)
        side_probability = (1 - self.probability) / 2
        self.draw_text(f"{self.probability:.0%} intended direction / {side_probability:.0%} each side", 640, 480, MUTED, self.tiny)

        self.draw_card(524, 116)
        self.draw_text(self.mode.upper(), 640, 538, MUTED, self.tiny)
        for index, line in enumerate(self.algorithm_lines):
            self.draw_text(line, 640, 560 + index * 22, TEXT, self.tiny)
        action = self.current_ai_action()
        next_action = action.title() if action else "--"
        self.draw_text(f"Next policy action: {next_action}   |   S: single step", 640, 610, ACCENT, self.tiny)

        can_play = self.policy is not None and not self.world.done
        label = "Pause [Space]" if self.playing else "Play [Space]"
        self.draw_button((620, 654, 140, 36), label, pygame.K_SPACE, self.playing, can_play)
        self.draw_button((770, 654, 128, 36), "Reset [R]", pygame.K_r)
        self.draw_button((908, 654, 148, 36), "New layout [N]", pygame.K_n)
        self.draw_card(704, 92)
        self.draw_message(self.message, 640, 718, 396)
        self.draw_text(self.last_move, 640, 775, MUTED, self.tiny)

    def draw(self):
        self.screen.fill(BACKGROUND)
        self.buttons = []
        self.draw_text("Warehouse robot", 24, 17, TEXT, self.title_font)
        self.draw_text("Collect a package. Deliver it. Return to the dock.", 24, 54, MUTED)
        self.draw_text(f"Layout seed: {self.world.seed_value}", 860, 25, MUTED)
        self.draw_text("Coordinates: (row, column)", 860, 53, MUTED, self.tiny)
        self.draw_button((24, 88, 140, 38), "Manual [M]", pygame.K_m, self.mode == "Manual")
        self.draw_button((174, 88, 190, 38), "Value Iteration [V]", pygame.K_v, self.mode == "Value Iteration")
        self.draw_button((374, 88, 210, 38), "Train Q-Learning [Q]", pygame.K_q, self.mode == "Q-Learning")
        self.draw_text("WAREHOUSE", 44, 135, MUTED, self.tiny)
        arrow_label = "Hide policy arrows [H]" if self.show_policy else "Show policy arrows [H]"
        self.draw_button((392, 133, 192, 22), arrow_label, pygame.K_h, self.show_policy, self.policy is not None)
        self.draw_grid()
        legend = [("P  Pickup", CELL_COLORS[PICKUP]), ("D  Packing", CELL_COLORS[PACKING]),
                  ("C  Dock", CELL_COLORS[DOCK]), ("Shelf / blocked", CELL_COLORS[SHELF]),
                  ("!  Congestion (-10)", CELL_COLORS[CONGESTION]), ("R  Robot", ROBOT_COLOR)]
        for index, (label, color) in enumerate(legend):
            x = 44 + (index % 3) * 184
            y = 730 + (index // 3) * 29
            pygame.draw.rect(self.screen, color, (x, y, 16, 16), border_radius=3)
            self.draw_text(label, x + 24, y - 1, MUTED, self.tiny)
        self.draw_text("Arrow keys: move   |   +/-: speed   |   Esc: quit", 44, 793, MUTED, self.tiny)
        self.draw_panel()
        self.present()

    def present(self):
        viewport = self.screen.get_rect().fit(self.window.get_rect())
        self.window.fill(BACKGROUND)
        if viewport.width == 0 or viewport.height == 0:
            return
        frame = self.screen
        if viewport.size != self.screen.get_size():
            frame = pygame.transform.smoothscale(self.screen, viewport.size)
        self.window.blit(frame, viewport)

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
            now = pygame.time.get_ticks()
            if self.playing and now - self.last_move_time >= self.delay_ms:
                action = self.current_ai_action()
                if action is None:
                    self.playing = False
                else:
                    self.step_robot(action)
                self.last_move_time = now
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()


if __name__ == "__main__":
    WarehouseRobotApp().run()
