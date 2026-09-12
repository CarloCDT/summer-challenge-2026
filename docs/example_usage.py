import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from railroad_env import RailroadGymEnv


def example_random_agent():
    env = RailroadGymEnv(
        width=25,
        height=17,
        num_towns=8,
        opponent_strategy="random",
        render_mode="human",
        seed=42,
    )

    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Game info: {info}")

    total_reward = 0
    for step in range(100):
        actions = {
            "actions": [
                (1, (np.random.randint(0, env.width), np.random.randint(0, env.height))),
                (2, (np.random.randint(0, 50), 0)),
            ]
        }

        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward

        print(f"Step {step + 1}: reward={reward}, scores={info['scores']}")

        if terminated:
            print(f"Episode finished! Total reward: {total_reward}")
            break

    env.close()


def example_action_format():
    env = RailroadGymEnv(width=25, height=17, num_towns=8, seed=42)
    obs, info = env.reset()

    actions = {
        "actions": [
            (0, (0, 0)),
            (1, (5, 10)),
            (2, (3, 0)),
            (1, (6, 11)),
        ]
    }

    obs, reward, terminated, truncated, info = env.step(actions)
    print(f"Reward: {reward}")
    print(f"Scores: {info['scores']}")


def example_custom_opponent():
    def my_custom_strategy(game_state):
        actions = []

        for y in range(5):
            for x in range(5):
                if game_state.paint_points[1] >= 1 and game_state.tracks[y, x] == -1:
                    actions.append(("PLACE", x, y))
                    break

        return actions if actions else [("WAIT",)]

    env = RailroadGymEnv(width=25, height=17, num_towns=8, seed=42)
    obs, info = env.reset()
    env.set_opponent_behavior(my_custom_strategy)

    for _ in range(10):
        actions = {"actions": [(0, (0, 0))]}
        obs, reward, terminated, truncated, info = env.step(actions)


if __name__ == "__main__":
    print("Example 1: Random agent with rendering")
    print("Note: Comment out render_mode='human' if you don't have a display")
    print()

    print("Example 2: Action format demonstration")
    example_action_format()
    print()

    print("Example 3: Custom opponent strategy")
    example_custom_opponent()
