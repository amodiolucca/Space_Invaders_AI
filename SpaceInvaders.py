import gym
import random
random.seed(0)

def main():
    env = gym.make('SpaceInvaders-v0', render_mode = "human")
    env.seed(0)
    numberOfSteps = 2
    env.reset()
    for i in range(numberOfSteps):
        episode_reward = 0
        while True:
            action = env.action_space.sample()
            new_obs, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                print("Reward of the iteration number {}: {} ".format(i+1, episode_reward))
                env.reset()
                break
    

main()
