import numpy as np
import pygame
from smb2_gym import SuperMarioBros2Env
from smb2_gym.app import InitConfig

class SuperMarioEnv:
    def __init__(self):
        config = InitConfig(level="1-1", character="mario")
        self.env = SuperMarioBros2Env(
            init_config=config,
            render_mode="human",
            action_type="simple"
        )
        self.done = False
        self.state = self.reset() 
        self.action_size = self.env.action_space.n
        self.obs_shape = self.env.observation_space.shape
        self.state_size = np.prod(self.obs_shape)
        
        
    def reset(self):
        obs, info = self.env.reset()
        state = self.obs_to_state_vector(obs) 
        self.env.render()
        return state
        # returns state after reset
    
    def step(self, action_index):
        obs, build_in_reward, self.done, truncated, info = self.env.step(action_index)
        reward = build_in_reward
        self.env.render()
        next_state = self.obs_to_state_vector(obs)
        self.state = next_state
        print(f"action: {action_index}, reward: {reward}") # to debug and see whats going on 
        return next_state, reward, self.done
    
    # flattens the 3 dimensional input from the image in a 1 dimensional vector also normalized from 0 to 255 to 0 to 1
    def obs_to_state_vector(self, obs):
        return obs.flatten().astype(np.float32) / 255.0
    
    
def test_env():
    env = SuperMarioEnv()

    print("Environment erfolgreich initialisiert!")
    print(f"Action Space Size: {env.action_size}")
    print(f"Observation Shape (flach): {env.state_size}\n")

    # Test reset
    state = env.reset()
    print("Reset erfolgreich!")
    print(f"State-Typ: {type(state)}, Länge: {len(state)}")

    # Test 1000 zufällige Schritte
    for i in range(1000):
        action = np.random.randint(env.action_size)
    
        next_state, reward, done = env.step(action)

        print(f"Step {i} | Action: {action} | Reward: {reward} | Done: {done}")

        if done:
            print("Episode zu Ende – führe Reset durch.")
            env.reset()
            break

    print("\nEnvironment-Test abgeschlossen!")
    

# to do: adjust keys to more sensable layout and action
# 0 = nothing 
# 1 = right
# 2 = left
# 3 = enter door
# 4 = jump
# 5 = nothing  
# 6 = right + jump
# 7 = left + jump
# 8 = right
# 9 = left
# 10 = duck
# 11 = duck + jump

def get_action(keys):
    
    # I don't know which action is actually doing nothing, 0 is just a quess
    ACTION_NOTHING = 0

    if keys[pygame.K_0]:
        return 0
    if keys[pygame.K_1]:
        return 1
    if keys[pygame.K_2]:
        return 2
    if keys[pygame.K_3]:
        return 3
    if keys[pygame.K_4]:
        return 4
    if keys[pygame.K_5]:
        return 5
    if keys[pygame.K_6]:
        return 6
    if keys[pygame.K_7]:
        return 7
    if keys[pygame.K_8]:
        return 8
    if keys[pygame.K_9]:
        return 9
    if keys[pygame.K_a]:
        return 10
    if keys[pygame.K_s]:
        return 11
    
    
    return ACTION_NOTHING


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

# to test enviroment:
# test_env()
    
# to play mario: 
# play_mario()