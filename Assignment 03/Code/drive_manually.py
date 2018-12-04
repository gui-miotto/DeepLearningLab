from __future__ import print_function

import argparse
from pyglet.window import key
import gym
import numpy as np
import pickle
import os
from datetime import datetime
import gzip
import json

import copy


'''
def key_press(k, mod):
    global restart
    #if k == 0xff0d: restart = True
    if k == key.ESCAPE: restart = True
    if k == key.LEFT:  a[0] = -1.0
    if k == key.RIGHT: a[0] = +1.0
    if k == key.UP:    a[1] = +1.0
    if k == key.DOWN:  a[2] = +0.4  # stronger brakes
'''

def key_press(k, mod):
    global restart
    #if k == 0xff0d: restart = True
    if k == key.ESCAPE: restart = True
    if k == key.UP:    
        a[3] = +1.0
        if a[0] == 0.0:
            a[1] = +1.0
    if k == key.LEFT:  
        a[0] = -1.0
        a[1] =  0.0  # Cut gas while turning
    if k == key.RIGHT: 
        a[0] = +1.0
        a[1] =  0.0  # Cut gas while turning
    if k == key.DOWN:  
        a[2] = +0.4  # stronger brakes

def key_release(k, mod):
    if k == key.LEFT and a[0] == -1.0: 
        a[0] = 0.0
        if a[3] == 1.0:
            a[1] = 1.0
    if k == key.RIGHT and a[0] == +1.0: 
        a[0] = 0.0
        if a[3] == 1.0:
            a[1] = 1.0
    if k == key.UP:    
        a[1] = 0.0
        a[3] = 0.0
    if k == key.DOWN:  
        a[2] = 0.0


def store_data(data, datasets_dir="./data"):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
    f = gzip.open(data_file,'wb')
    pickle.dump(data, f)


def save_results(episode_rewards, results_dir="./results"):
    # save results
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

     # save statistics in a dictionary and write them into a .json file
    results = dict()
    results["number_episodes"] = len(episode_rewards)
    results["episode_rewards"] = episode_rewards

    results["mean_all_episodes"] = np.array(episode_rewards).mean()
    results["std_all_episodes"] = np.array(episode_rewards).std()
 
    fname = os.path.join(results_dir, "results_manually-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))
    fh = open(fname, "w")
    json.dump(results, fh)
    print('... finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect_data", action="store_true", default=True, help="Collect the data in a pickle file.")
    args = parser.parse_args()
    
    good_samples = {
        "state": [],
        "next_state": [],
        "reward": [],
        "action": [],
        "terminal" : [],
    }
    episode_samples = copy.deepcopy(good_samples)

    env = gym.make('CarRacing-v0').unwrapped
    env.reset()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    a = np.zeros(4, dtype=np.float32)
    
    episode_rewards = []
    good_steps = episode_steps = 0
    # Episode loop
    while True:
        episode_samples["state"] = []
        episode_samples["action"] = []
        episode_samples["next_state"] = []
        episode_samples["reward"] = []
        episode_samples["terminal"] = []
        episode_reward = 0
        state = env.reset()
        restart = False
        episode_steps = good_steps
        # State loop
        while True:
            next_state, r, done, info = env.step(a[:3])
            episode_reward += r

            episode_samples["state"].append(state)            # state has shape (96, 96, 3)
            episode_samples["action"].append(np.array(a[:3]))     # action has shape (1, 3)
            episode_samples["next_state"].append(next_state)
            episode_samples["reward"].append(r)
            episode_samples["terminal"].append(done)
            
            state = next_state
            episode_steps += 1

            if episode_steps % 1000 == 0 or done:
                print("\nstep {}".format(episode_steps))

            env.render()
            if done or restart: 
                break
        
        if not restart:
            good_steps = episode_steps

            episode_rewards.append(episode_reward)
            
            good_samples["state"].append(episode_samples["state"])
            good_samples["action"].append(episode_samples["action"])
            good_samples["next_state"].append(episode_samples["next_state"])
            good_samples["reward"].append(episode_samples["reward"])
            good_samples["terminal"].append(episode_samples["terminal"])

            print('... saving data')
            store_data(good_samples, "./data")
            save_results(episode_rewards, "./results")

    env.close()

    

   
