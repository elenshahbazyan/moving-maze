import yaml
import os
from gridworld_maze import GridWorldEnv  # Adjust this import path based on your file structure


def load_and_test_environment(config_file="config.yaml"):
    """Load environment from YAML config and test it"""

    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"Config file '{config_file}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Available files: {os.listdir('.')}")
        return None

    # Load YAML configuration
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Extract environment configuration
    env_config = config.get('env', {})
    portal_config = env_config.get('portals', {})

    print("=== Loading Environment ===")
    print(f"Size: {env_config.get('size', 20)}")
    print(f"Portal pairs: {env_config.get('n_portal_pairs', 0)}")
    print(f"Closer portals: {portal_config.get('closer', False)}")
    print(f"Farther portals: {portal_config.get('farther', False)}")
    print(f"Render mode: {env_config.get('render_mode', None)}")
    print("=============================")

    # Create environment with portal flags from YAML
    env = GridWorldEnv(
        size=env_config.get('size', 20),
        n_portal_pairs=env_config.get('n_portal_pairs', 0),
        portal_flags={
            'closer': portal_config.get('closer', False),
            'farther': portal_config.get('farther', False)
        },
        wall_update_freq=env_config.get('wall_update_freq', 3),
        render_mode=env_config.get('render_mode', None)
    )

    print(f"Environment created successfully!")
    print(f"Actual portal flags: {env.portal_flags}")
    print(f"Actual portal pairs: {env.n_portal_pairs}")
    print(f"Total portals in environment: {len(env._portals)}")

    return env


def test_environment(env, num_steps=20):
    """Test the environment with random actions"""

    print("\n=== Testing Environment ===")

    # Reset environment
    obs, info = env.reset()
    print(f"Environment reset!")
    print(f"Agent position: {obs['agent']}")
    print(f"Target position: {obs['target']}")
    print(f"Distance to target: {info['distance']}")

    if len(env._portals) > 0:
        print(f"Portals in environment:")
        for i, portal in enumerate(env._portals):
            print(f"  Portal {i + 1}: {portal['type']} at {portal['pos']}")
    else:
        print("No portals in environment")

    # Test some steps
    print(f"\nRunning {num_steps} test steps...")
    for step in range(num_steps):
        # Random action
        action = env.action_space.sample()
        action_names = ["right", "up", "left", "down", "wait"]

        obs, reward, done, truncated, info = env.step(action)

        print(f"Step {step + 1:2d}: {action_names[action]:5s} -> "
              f"pos={obs['agent']}, reward={reward:6.3f}, distance={info['distance']:4.1f}")

        # Check for special events
        if info.get('used_portal', False):
            print(f"         Used {info['portal_type']} portal!")
        if info.get('hit_wall', False):
            print(f"         Hit a wall!")
        if info.get('goal_reached', False):
            print(f"         🎉 GOAL REACHED! 🎉")
            break
        if truncated:
            print(f"         Episode truncated!")
            break

    print("=========================")


def interactive_test(env):
    """Interactive testing - control agent manually"""

    print("\n=== Interactive Mode ===")
    print("Controls: w=up, s=down, a=left, d=right, space=wait, q=quit")
    print("========================")

    obs, info = env.reset()

    while True:
        print(f"\nAgent at {obs['agent']}, Target at {obs['target']}, Distance: {info['distance']:.1f}")

        # Get user input
        try:
            key = input("Enter action (w/a/s/d/space/q): ").strip().lower()
        except KeyboardInterrupt:
            print("\nExiting...")
            break

        if key == 'q':
            break

        # Convert key to action
        action_map = {'w': 1, 'a': 2, 's': 3, 'd': 0, ' ': 4, 'space': 4}
        if key not in action_map:
            print("Invalid key! Use w/a/s/d/space/q")
            continue

        action = action_map[key]
        obs, reward, done, truncated, info = env.step(action)

        print(f"Reward: {reward:.3f}")
        if info.get('used_portal', False):
            print(f"Used {info['portal_type']} portal!")
        if info.get('hit_wall', False):
            print("Hit a wall!")

        if done:
            print("🎉 Goal reached! 🎉")
            break
        if truncated:
            print("Episode truncated!")
            break


if __name__ == "__main__":
    # Load and test environment
    env = load_and_test_environment("config.yaml")

    if env is not None:
        # Automatic test
        test_environment(env, num_steps=15)

        # Ask if user wants interactive mode
        try:
            response = input("\nDo you want to try interactive mode? (y/n): ").strip().lower()
            if response == 'y':
                interactive_test(env)
        except KeyboardInterrupt:
            print("\nSkipping interactive mode...")

        # Close environment
        env.close()
        print("Environment closed.")
