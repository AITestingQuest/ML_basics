# Import the gymnasium library
import gymnasium as gym

# Create the environment
env = gym.make('MountainCar', render_mode='rgb_array')

# Get the initial state
initial_state, info = env.reset(seed=42)

position = initial_state[0]
velocity = initial_state[1]

print(f"The position of the car along the x-axis is {position} (m)")
print(f"The velocity of the car is {velocity} (m/s)")