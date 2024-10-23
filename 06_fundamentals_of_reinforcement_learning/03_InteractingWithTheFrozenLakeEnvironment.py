import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', render_mode='rgb_array')
initial_state, _ = env.reset()

# Complete the render function
def render():
    state_image = env.render()
    plt.imshow(state_image)
    plt.show()

# Define the sequence of actions
actions = [1,1,2,2,1,2]

for action in actions: 
  # Execute each action
  state, reward, terminated, _, _ = env.step(action)
  # Render the environment
  render()
  if terminated:
  	print("You reached the goal!")