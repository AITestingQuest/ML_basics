import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('MountainCar', render_mode='rgb_array')
initial_state, _ = env.reset()

# Complete the render function
def render():
    state_image = env.render()
    plt.imshow(state_image)
    plt.show()

# Call the render function    
render()