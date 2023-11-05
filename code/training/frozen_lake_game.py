import numpy as np
import gym
import random
import time
from IPython.display import clear_output

# create environment
env = gym.make("FrozenLake-v1")

# init q table
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))
print(q_table)

# parameters
num_episodes = 10000
max_steps = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []
def reinforcement_learning_with_params():
    # q learning algorithm
    for episode in range(num_episodes):
        # initialize new episode params
        state = env.reset()[0]
        done = False
        rewards_current_episode = 0

        for step in range(max_steps): 
            # Exploration-exploitation trade-off
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_table[state,:]) 
            else:
                action = env.action_space.sample()

            # Take new action
            new_state, reward, done, truncated, info = env.step(action)

            # Update Q-table
            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

            # Set new state
            state = new_state
            
            # Add new reward
            rewards_current_episode += reward
            if done == True: 
                break

        # Exploration rate decay
        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
        # Add current episode reward to total rewards list
        rewards_all_episodes.append(rewards_current_episode)

    # Calculate and print the average reward per thousand episodes
    rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
    count = 1000

    print("********Average reward per thousand episodes********\n")
    for r in rewards_per_thousand_episodes:
        print(count, ": ", str(sum(r/1000)))
        count += 1000

    # Print updated Q-table
    print("\n\n********Q-table********\n")
    print(q_table)

def watch_algorithm():
    # Watch our agent play Frozen Lake by playing the best action 
    # from each state according to the Q-table

    for episode in range(3):
        # initialize new episode params
        state = env.reset()[0]
        done = False
        print("*****EPISODE ", episode+1, "*****\n\n\n\n")
        time.sleep(1)

        for step in range(max_steps):        
            # Show current state of environment on screen
            clear_output(wait=True)
            print(env.render())
            time.sleep(0.3)
            # Choose action with highest Q-value for current state
            action = np.argmax(q_table[state,:])        
            new_state, reward, done, truncated, info = env.step(action) 

            # Take new action
            if done:
                clear_output(wait=True)
                print(env.render())
                if reward == 1:
                    # Agent reached the goal and won episode
                        print("****You reached the goal!****")
                        time.sleep(3)
                else:
                    # Agent stepped in a hole and lost episode
                    print("****You fell through a hole!****")
                    time.sleep(3)
                    clear_output(wait=True)

            # Set new state
            state = new_state

    env.close()

watch_algorithm()