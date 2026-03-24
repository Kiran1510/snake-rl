"""
Rendering backends for the Snake environment.

AsciiRenderer: Terminal-based, always available.
PygameRenderer: GUI-based, requires pygame (import-guarded).
"""

import sys
import time
from typing import Optional

from snake_rl.env.snake_env import SnakeEnv


class AsciiRenderer:
    """
    Terminal-based renderer for the Snake environment.

    Prints the game grid to stdout with optional frame delay.
    Works in any terminal, no dependencies.
    """

    def __init__(self, clear_screen: bool = True, delay: float = 0.1):
        """
        Parameters
        ----------
        clear_screen : bool
            Whether to clear terminal between frames.
        delay : float
            Seconds to wait between frames.
        """
        self.clear_screen = clear_screen
        self.delay = delay

    def render(self, env: SnakeEnv) -> None:
        """Render the current environment state to the terminal."""
        if self.clear_screen:
            # ANSI escape to clear screen and move cursor to top-left
            sys.stdout.write("\033[2J\033[H")
        print(env.render_ascii())
        if self.delay > 0:
            time.sleep(self.delay)

    def close(self) -> None:
        """No resources to clean up for ASCII rendering."""
        pass


class PygameRenderer:
    """
    Pygame-based GUI renderer for the Snake environment.

    Renders the game as a graphical window with colored cells.
    Requires pygame to be installed.

    Usage
    -----
    >>> renderer = PygameRenderer(cell_size=30)
    >>> renderer.render(env)
    >>> # ... game loop ...
    >>> renderer.close()
    """

    # Color scheme
    COLOR_BG = (20, 20, 20)
    COLOR_GRID = (40, 40, 40)
    COLOR_SNAKE_HEAD = (0, 200, 80)
    COLOR_SNAKE_BODY = (0, 150, 60)
    COLOR_FOOD = (220, 50, 50)
    COLOR_TEXT = (220, 220, 220)

    def __init__(self, cell_size: int = 30, fps: int = 10):
        """
        Parameters
        ----------
        cell_size : int
            Pixel size of each grid cell.
        fps : int
            Target frames per second.
        """
        self.cell_size = cell_size
        self.fps = fps
        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None
        self._initialized = False

    def _init_pygame(self, grid_size: int) -> None:
        """Lazy-initialize pygame on first render call."""
        try:
            import pygame
            self._pygame = pygame
        except ImportError:
            raise ImportError(
                "pygame is required for PygameRenderer. "
                "Install it with: pip install pygame"
            )

        pygame.init()
        window_w = grid_size * self.cell_size
        window_h = grid_size * self.cell_size + 40  # extra space for score bar
        self._screen = pygame.display.set_mode((window_w, window_h))
        pygame.display.set_caption("Snake RL")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("monospace", 18)
        self._initialized = True

    def render(self, env: SnakeEnv) -> bool:
        """
        Render the current environment state.

        Returns
        -------
        bool
            False if the user closed the window (signals to stop),
            True otherwise.
        """
        if not self._initialized:
            self._init_pygame(env.grid_size)

        pygame = self._pygame

        # Handle pygame events (window close, etc.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False

        # Clear screen
        self._screen.fill(self.COLOR_BG)

        cs = self.cell_size

        # Draw grid lines
        for x in range(env.grid_size + 1):
            pygame.draw.line(
                self._screen, self.COLOR_GRID,
                (x * cs, 0), (x * cs, env.grid_size * cs)
            )
        for y in range(env.grid_size + 1):
            pygame.draw.line(
                self._screen, self.COLOR_GRID,
                (0, y * cs), (env.grid_size * cs, y * cs)
            )

        # Draw snake body (from tail to head so head is on top)
        for i in range(len(env.snake) - 1, -1, -1):
            sx, sy = env.snake[i]
            color = self.COLOR_SNAKE_HEAD if i == 0 else self.COLOR_SNAKE_BODY
            rect = pygame.Rect(sx * cs + 1, sy * cs + 1, cs - 2, cs - 2)
            pygame.draw.rect(self._screen, color, rect)

            # Draw eyes on the head
            if i == 0:
                self._draw_eyes(sx, sy, env.direction)

        # Draw food
        fx, fy = env.food
        if 0 <= fx < env.grid_size and 0 <= fy < env.grid_size:
            food_rect = pygame.Rect(fx * cs + 2, fy * cs + 2, cs - 4, cs - 4)
            pygame.draw.rect(self._screen, self.COLOR_FOOD, food_rect, border_radius=cs // 4)

        # Draw score bar
        score_y = env.grid_size * cs + 5
        score_text = self._font.render(
            f"Score: {env.score}   Steps: {env.steps}   Length: {len(env.snake)}",
            True, self.COLOR_TEXT
        )
        self._screen.blit(score_text, (10, score_y))

        pygame.display.flip()
        self._clock.tick(self.fps)

        return True

    def _draw_eyes(self, hx: int, hy: int, direction: tuple[int, int]) -> None:
        """Draw two small eye dots on the snake head based on heading."""
        pygame = self._pygame
        cs = self.cell_size
        cx = hx * cs + cs // 2
        cy = hy * cs + cs // 2
        eye_r = max(2, cs // 10)
        offset = cs // 5

        dx, dy = direction
        # Eyes are perpendicular to the direction of travel
        perp_x, perp_y = -dy, dx

        # Shift eyes forward in the direction of travel
        fwd = cs // 6
        eye1 = (cx + perp_x * offset + dx * fwd, cy + perp_y * offset + dy * fwd)
        eye2 = (cx - perp_x * offset + dx * fwd, cy - perp_y * offset + dy * fwd)

        pygame.draw.circle(self._screen, (255, 255, 255), eye1, eye_r)
        pygame.draw.circle(self._screen, (255, 255, 255), eye2, eye_r)

    def close(self) -> None:
        """Shut down pygame."""
        if self._pygame is not None and self._initialized:
            self._pygame.quit()
            self._initialized = False


def play_human(grid_size: int = 20, cell_size: int = 30, fps: int = 8) -> None:
    """
    Play Snake manually with arrow keys (requires pygame).

    This is useful for sanity-checking the environment behavior.
    Arrow keys control absolute direction; the environment converts
    to relative actions internally.
    """
    try:
        import pygame
    except ImportError:
        print("pygame required for human play. Install with: pip install pygame")
        return

    env = SnakeEnv(grid_size=grid_size)
    renderer = PygameRenderer(cell_size=cell_size, fps=fps)
    obs, _ = env.reset()
    renderer.render(env)

    # Map from absolute key direction to relative action
    running = True
    while running:
        action = Action.STRAIGHT  # default: keep going straight

        pygame.event.pump()
        keys = pygame.key.get_pressed()

        # Determine desired absolute direction from key press
        desired = None
        if keys[pygame.K_UP]:
            desired = (0, -1)
        elif keys[pygame.K_DOWN]:
            desired = (0, 1)
        elif keys[pygame.K_LEFT]:
            desired = (-1, 0)
        elif keys[pygame.K_RIGHT]:
            desired = (1, 0)

        if desired is not None:
            action = _abs_to_relative(env.direction, desired)

        obs, reward, terminated, truncated, info = env.step(action)
        alive = renderer.render(env)

        if not alive:
            break

        if terminated or truncated:
            print(f"Game over! Score: {env.score}, Cause: {info.get('cause', 'unknown')}")
            time.sleep(2)
            obs, _ = env.reset()

    renderer.close()


def _abs_to_relative(current_dir: tuple[int, int], desired_dir: tuple[int, int]) -> int:
    """Convert an absolute desired direction to a relative action given current heading."""
    from snake_rl.env.snake_env import _turn_left, _turn_right, Action

    if desired_dir == current_dir:
        return Action.STRAIGHT
    elif _turn_left(current_dir) == desired_dir:
        return Action.LEFT
    elif _turn_right(current_dir) == desired_dir:
        return Action.RIGHT
    else:
        # Reverse direction requested — just go straight (can't reverse)
        return Action.STRAIGHT


# Re-export Action for convenience in play_human
from snake_rl.env.snake_env import Action
