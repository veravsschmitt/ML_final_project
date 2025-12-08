import os
import glob
import torch
import json
from mario_env import SuperMarioEnv, play_mario
from agent_Linear import DQN_Linear_Agent
from agent_Conv import DQN_Conv_Agent
from datetime import datetime

def get_latest_model_path(models_dir, agent_type):
    pattern = os.path.join(models_dir, f"model_{agent_type}_*.pth")
    model_files = glob.glob(pattern)
    
    if not model_files:
        return None
    model_files.sort()
    return model_files[-1]

def play(agentType, rendering=True):
    # Environment & Agent setup
    if agentType == "DQN_LINEAR":
        env = SuperMarioEnv(rendering, threedim=False)
        agent = DQN_Linear_Agent(env.state_size, env.action_size)
    elif agentType == "DQN_CONV":
        env = SuperMarioEnv(rendering, threedim=True)
        agent = DQN_Conv_Agent(env.state_size, env.action_size)
    elif agentType == "MYSELF":
        play_mario()
        return
    else: 
        play_mario()
        return
    

    # Load latest model
    models_dir = "models"
    latest_model = get_latest_model_path(models_dir, agentType)
    if latest_model:
        agent.load(latest_model)
        # Load epsilon if available
        prefix = f"model_{agentType}_"
        meta_prefix = f"meta_{agentType}_"
        meta_path = latest_model.replace(prefix, meta_prefix).replace(".pth", ".json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                agent.epsilon = meta.get("epsilon", 0.0)  # For play, usually set to 0
        print(f"Loaded latest model: {latest_model}")
    else:
        print("No model found. Exiting...")
        return

    # Set epsilon to 0 for playing (no exploration)
    agent.epsilon = 0.0

    # Play a single episode
    state = env.reset()
    total_reward = 0
    done = False

    print("\nStarting episode...\n")
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward

    print("\nEpisode finished!")
    print(f"total_reward: {total_reward:.2f}")
    print(f"level_reached: {env.get_ep_level()}")
    print(f"world_reached: {env.get_ep_world()}")
    print(f"X_global_reached: {env.get_ep_stats_xglobal_reached()}")
    print(f"Actions_taken: {env.get_ep_stats_actions_taken()}")


if __name__ == "__main__":
    agentType = input("Whom would you like to play (MYSELF is default on invalid input)? (DQN_LINEAR / DQN_CONV / MYSELF) ").strip().upper()
    play(agentType)
