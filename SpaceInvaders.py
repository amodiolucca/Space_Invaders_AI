import gym
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import cv2
random.seed(0)

def main():
    #env = gym.make('SpaceInvaders-v0', render_mode = "human")
    env = gym.make('SpaceInvaders-v0')
    env.seed(0)
    numberOfSteps = 2
    env.reset()
    #Hyperparameters
    num_episodes = 100
    batch_size = 32
    buffer_size = 10000
    epsilon = 0.1
    gamma = 0.99


    experience_buffer = []

    #creating the model using keras with convolutional layers and dense layers
    model = keras.Sequential([
        keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4)),
        keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        keras.layers.Conv2D(64, (2, 2), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(6, activation='linear'),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
   

    for i in range(numberOfSteps):
        episode_reward = 0
        state = preprocess_observation(env.reset())
  
        while True:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Aleatory action with epsilon probability
            else:
                action = np.argmax(model.predict(np.expand_dims(state, axis=0)))

            next_state, reward, done, info = env.step(action)
            next_state = preprocess_observation(next_state)

            experience_buffer.append((state, action, reward, next_state, done))

            episode_reward += reward
            state = next_state

            if done:
                print("Reward of the episode number {}: {} ".format(i+1, episode_reward))
                env.reset()
                break
    


def preprocess_observation(observation):
    #Convert to gray scale
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    # Resize to 84x84
    observation = cv2.resize(observation, (84, 84))
    return observation


def update_dqn(model, minibatch, gamma):
    states = []
    targets = []
    
    for state, action, reward, next_state, done in minibatch:
        
        #Q funcition to current state
        q_values = model.predict(state[np.newaxis])
        
        # Q function to next state
        next_q_values = model.predict(next_state[np.newaxis])
        
        if done:
            # Q function equals reward
            q_values[0][action] = reward
        else:
            # Refreshing Q function value
            q_values[0][action] = reward + gamma * np.max(next_q_values)
        
        states.append(state)
        targets.append(q_values)
    
    states = np.vstack(states)
    targets = np.vstack(targets)
    
    # Training the neural network
    model.fit(states, targets, epochs=1, verbose=0)

main()
