import time
import numpy as np
import pygame
from smb2_gym import SuperMarioBros2Env
from smb2_gym.app import InitConfig


class SuperMarioEnv:
    def __init__(self, rendering):
        self.done = False
        self.rendering = rendering
        self.state = self.reset() 
        self.action_size = self.env.action_space.n
        self.obs_shape = self.env.observation_space.shape
        self.state_size = np.prod(self.obs_shape)

    
    def reset(self):
        config = InitConfig(level="1-2", character="mario")
        self.env = SuperMarioBros2Env(
            init_config=config,
            render_mode="human" if self.rendering else None,
            action_type="simple"
        )
        
        # Game Metrics Tracked
        self.ep_stats_starttime = time.time()
        self.ep_stats_actions_taken = 0 
        self.ep_xglobal_reached = 0
          
          
        
        obs, info = self.env.reset()
        self.info_x_global = info['pos'].x_global
        (world, level) = parse_level(info['game'].level)
        self.info_level = level
        self.info_world = world
        
        self.info_hearts = info['pc'].hearts
        self.info_cherries = info['pc'].cherries
        
        
        state = self.obs_to_state_vector(obs) 
        if self.rendering == True:
            self.env.render()
        self.done = False
        return state
        # returns state after passreset
    
    def step(self, action_index):
        obs, build_in_reward, self.done, truncated, next_info = self.env.step(action_index)
        reward = get_reward(self.info_x_global, self.info_level, self.info_world, self.info_hearts, self.info_cherries, next_info)
        if self.done == True:
            end_time = time.time()
            self.ep_stats_total_time = end_time - self.ep_stats_starttime
        if self.rendering == True:
            self.env.render()
        next_state = self.obs_to_state_vector(obs)
        self.state = next_state
        
        self.info_x_global = next_info['pos'].x_global
        (world, level) = parse_level(next_info['game'].level)
        self.info_level = level
        self.info_world = world
        self.info_hearts = next_info['pc'].hearts
        self.info_cherries = next_info['pc'].cherries
        
        self.ep_stats_actions_taken += 1
        self.ep_xglobal_reached = max(self.ep_xglobal_reached, next_info['pos'].x_global)
        print(f"action: {action_index}, reward: {reward}") # to debug and see whats going on 
        return next_state, reward, self.done
    
    def get_ep_stats_actions_taken(self):
        return self.ep_stats_actions_taken
    
    def get_ep_stats_xglobal_reached(self):
        return self.ep_xglobal_reached
    
    def get_ep_total_time(self):
        return self.ep_stats_total_time
    
    def get_ep_level(self):
        return self.info_level
    
    def get_ep_world(self):
        return self.info_world
    
    # flattens the 3 dimensional input from the image in a 1 dimensional vector also normalized from 0 to 255 to 0 to 1
    def obs_to_state_vector(self, obs):
        return obs.flatten().astype(np.float32) / 255.0
        
    
    
def test_env():
    env = SuperMarioEnv(True)

    print("Environment installed!")
    print(f"Action Space Size: {env.action_size}")
    print(f"Observation Shape (flattened): {env.state_size}\n")

    # Test reset
    state = env.reset()
    print("Reset successful")
    print(f"State-Typ: {type(state)}, Length: {len(state)}")

    # Test 1000 zufällige Schritte
    for i in range(1000):
        action = np.random.randint(env.action_size)
    
        next_state, reward, done = env.step(action)

        print(f"Step {i} | Action: {action} | Reward: {reward} | Done: {done}")

        if done:
            print("Episode zu Ende – führe Reset durch.")
            env.reset()
            break

    print("\nEnvironment-Test completed")
    

def get_action(keys):
    
    # Action mapping (discovered through trying):
    #
    # 0: Nothing
    # 1: Right
    # 2: Left
    # 3: Up (Enter door)
    # 4: A Button (Jump)
    # 5: B Button (Pickup/Throw)
    # 6: Right + A
    # 7: Left + A
    # 8: Right + B
    # 9: Left + B
    # 10: Down (Duck)
    # 11: Down + A

    # Directional keys
    right = keys[pygame.K_RIGHT]
    left = keys[pygame.K_LEFT]
    up = keys[pygame.K_UP]
    down = keys[pygame.K_DOWN]

    # Action buttons
    jump = keys[pygame.K_a]      # A Button
    pickup = keys[pygame.K_s]    # B Button

    # Check combinations
    if right and jump:
        return 6
    if left and jump:
        return 7
    if right and pickup:
        return 8
    if left and pickup:
        return 9
    if down and jump:
        return 11
    if down:
        return 10
    if up:
        return 3
    if jump:
        return 4
    if pickup:
        return 5
    if right:
        return 1
    if left:
        return 2

    return 0 # nothing action


def play_mario():
    # Mario enviroment
    config = InitConfig(level="1-1", character="mario")
    env = SuperMarioBros2Env(
            init_config=config,
            render_mode="human",
            action_type="simple"
        )

    obs, info = env.reset()

    # initialize pygame
    pygame.init()
    pygame.display.set_caption("Super Mario Bros 2 – playable")

    running = True
    clock = pygame.time.Clock()

    while running:
        clock.tick(60)  
        
        # check for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        # esc to end
        if keys[pygame.K_ESCAPE]:
            running = False


        action = get_action(keys)

        # do the step
        obs, reward, done, truncated, info = env.step(action)
        env.render()   # udpate window

        if done:
            env.reset()

    env.close()
    pygame.quit()
    

def get_reward(x_global, level, world, hearts, cherries, new_info):
    
    reward = 0



    reward += (new_info['pc'].hearts - hearts)* (50)       # plus points for hearts
    print(f"+ hearts:{reward}")
    reward += (new_info['pc'].cherries - cherries) * (3)   # plus points for collecting cherries
    print(f"+ cherries:{reward}")
    

   
    (new_world, new_level) = parse_level(new_info['game'].level)
    
    reward += (level - new_level) * (-20)    # plus points for finishing a level
    reward += (world - new_world) * (-200)   # plus points for finishing a world (! high enough to counter the minus points from "losing" teh levels)
    print(f"+ level + world:{reward}")

    # minus points for losing lives
    if new_info.get('life_lost'):
        reward -= 100
        print(f"+ lifelost:{reward}")
    else:
        if level == new_level:
            if world == new_world: 
                global_x_reward= (new_info['pos'].x_global - x_global) * (10) # plus points for each pixel more to the right (end of the level)
                if abs(global_x_reward) < 100:
                    reward += global_x_reward
                else:
                    print("global x difference to big")

                print(f"+ no live loss x:{reward}")

    if new_info['game'].is_game_over == False:
        reward += 0.5     # small satying alive bonus
    else: 
        reward -= 50

    print(f"+ stayalive/gameover:{reward}")
            
    return reward

def parse_level(lvl):
    if isinstance(lvl, str) and "-" in lvl:
        w, s = lvl.split("-")
        return int(w), int(s)
    return (0, 0)

# to test enviroment:
# test_env()
    
# to play mario: 
# play_mario()