# Training of an agent (DQN implemented)
# base from by https://github.com/krazyness/CRBot-public/blob/main/env.py and topics in CS project

import numpy as np
import os
import torch
import glob
import json
from mario_env import SuperMarioEnv
from agent import DQNAgent
from pynput import keyboard
from datetime import datetime

# For converting data to python built-ins
def convert_to_jsonable(obj):
    """Recursively convert NumPy types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: convert_to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_jsonable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# keyboard controller to stop training in a save way (press q to stop the training)
class KeyboardController:
    def __init__(self):
        self.should_exit = False
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            if key.char == 'q':
                print("\nShutdown requested - cleaning up...")
                self.should_exit = True
        # just ignore if a special key is pressed which is not a char value
        except AttributeError:
            pass  # Special key pressed
            
    def is_exit_requested(self):
        return self.should_exit

# if there is at least one model already in the model folder, get the latest one
def get_latest_model_path(models_dir):
    model_files = glob.glob(os.path.join(models_dir, "model_*.pth"))
    if not model_files:
        return None
    model_files.sort()  # Lexicographical sort works for timestamps
    return model_files[-1]

# accepted inputs for agentType:  DQN or PPO  (default for invalid values is DQN)
def train(agentType):

    env = SuperMarioEnv()

    if agentType == "DQN":
        agent = DQNAgent(env.state_size, env.action_size)
    elif agentType == "PPO":
        # to do: implement PPO case
        pass
    else: 
        agent = DQNAgent(env.state_size, env.action_size)

    # currently just placeholder for a version with different personas
    # Ensure models directory exists
    persona = "survival" 
    models_dir = os.path.join("models", persona)
    os.makedirs(models_dir, exist_ok=True)


    # Load latest model if available
    latest_model = get_latest_model_path(models_dir)
    if latest_model:
        agent.load(latest_model)
        # Load epsilon
        meta_path = latest_model.replace("model_", "meta_").replace(".pth", ".json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                agent.epsilon = meta.get("epsilon", 1.0)
            print(f"Epsilon loaded: {agent.epsilon}")

    # to do: set constants for nicer manipulation of episodes batch size and so on 
    controller = KeyboardController()
    episodes = 10000
    batch_size = 32

    # data tracking
    metrics_path = os.path.join(models_dir, "training_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception:
            metrics = []
    else:
        metrics = []

    try:
        for ep in range(episodes):
            if controller.is_exit_requested():
                print("Training interrupted by user.")
                break

            state = env.reset()
            total_reward = 0

            print(f"Episode {ep + 1} starting. Epsilon: {agent.epsilon:.3f}") 
            
            done = False
            while not done:

                # get current action the model would do 
                action = agent.act(state)
                # perform the action and save the outcome
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                # update the model
                agent.replay(batch_size)
                # update vars
                state = next_state
                total_reward += reward
            print(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}")

            # appending the data
            metrics.append({
                "episode": ep + 1,
                "persona": persona,
                "total_reward": total_reward,
                "epsilon": round(agent.epsilon, 4),
                "time_alive": env.episode_stats.get("time_alive", 0),
                "troops_deployed": env.episode_stats.get("troops_deployed", 0),
                "average_elixir": env.episode_stats.get("average_elixir", 0),
                "actions_taken": env.episode_stats.get("actions_taken", 0),
                "rank": env.episode_stats.get("rank", "unknown"),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            # Save model and epsilon every 10 episodes
            if ep % 10 == 0:
                # update the values in the target model with those from the model we are training on
                agent.update_target_model()
                
                # save model 
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(models_dir, f"model_{timestamp}.pth")
                torch.save(agent.model.state_dict(), model_path)

                # save epsilon value as meta data
                meta_path = os.path.join(models_dir, f"meta_{timestamp}.json")
                with open(meta_path, "w") as f:
                    json.dump({"epsilon": agent.epsilon}, f)

                # saving metrics
                with open(metrics_path, "w") as f:
                    json.dump(convert_to_jsonable(metrics), f, indent=4)
                    
                print(f"Model and epsilon saved: {model_path}")
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Ctrl+C detected — saving current progress...")

    finally:
        # Final save on exit
        agent.update_target_model()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(models_dir, f"model_{timestamp}.pth")
        torch.save(agent.model.state_dict(), model_path)

        meta_path = os.path.join(models_dir, f"meta_{timestamp}.json")
        with open(meta_path, "w") as f:
            json.dump({"epsilon": agent.epsilon}, f)

        with open(metrics_path, "w") as f:
            json.dump(convert_to_jsonable(metrics), f, indent=4)

        print(f"[FINAL SAVE] Model and metrics saved safely at {model_path}")

if __name__ == "__main__":
    
    agentType = input("Which agent would you like to train? (DQN / PPO) DQN will be trained if input invalid: ").strip().upper()
    # DQN right now, PPO needs to be added
    if (agentType == "DQN") :
        train("DQN")
    else:
        train("DQN")
        