# Third-party packages and modules:
from collections import deque
from datetime import datetime
import numpy as np
import gym, os, json
# My packages and modules:
from agent import Agent
import utils

def run_episode(env, agent, rendering=True, max_timesteps=1000):
    # Reset reward accumulator
    episode_reward = 0
    # Inform environment and agent that a new episode is about to begin:
    env_state = env.reset()
    agent.begin_new_episode(state0=env_state)

    for _ in range(max_timesteps):
        # Request action from agent:
        agent_action = agent.get_action(env_state)
        # Given this action, get the next environment state and reward:
        env_state, r, done, info = env.step(agent_action)   
        # Render the state screen:
        if rendering:
            env.render()
        # Accumulate reward
        episode_reward += r
        # Check if environment signaled the end of the episode:
        if done: break

    return episode_reward

def save_performance_results(episode_rewards, directory):
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
    
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{directory}results_bc_agent-{time_stamp}.json"
    with open(fname, "w") as fh:
        json.dump(results, fh)


if __name__ == "__main__":
    # Number of episodes to test:
    n_test_episodes = 15

    # Initialize environment and agent:
    env = gym.make('CarRacing-v0').unwrapped
    agent = Agent.from_file('saved_models/')

    # Episodes loop:
    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=True)
        episode_rewards.append(episode_reward)
        print(f'Episode {i+1} reward:{episode_reward:.2f}')
    env.close()

    # save reward statistics in a .json file
    save_performance_results(episode_rewards, 'performance_results/')
    print('... finished')
