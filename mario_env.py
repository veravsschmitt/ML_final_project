import numpy as np
import pygame
from smb2_gym import SuperMarioBros2Env
from smb2_gym.app import InitConfig

class SuperMarioEnv:
    def __init__(self):
        config = InitConfig(level="1-3", character="luigi")
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

    # Test 10 zufällige Schritte
    for i in range(400):
        action = np.random.randint(env.action_size)
        if i > 20:  # to test what the actions are
            action = 1

        next_state, reward, done = env.step(action)

        print(f"Step {i} | Action: {action} | Reward: {reward} | Done: {done}")

        if done:
            print("Episode zu Ende – führe Reset durch.")
            env.reset()
            break

    print("\nEnvironment-Test abgeschlossen!")
    
# to do: update these actions to the right ones and to all 12
ACTION_NOTHING = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_JUMP = 3
ACTION_LEFT_JUMP = 4
ACTION_RIGHT_JUMP = 5

# to do: update these actions to the right ones and to all 12
def get_action(keys):
    left = keys[pygame.K_LEFT]
    right = keys[pygame.K_RIGHT]
    jump = keys[pygame.K_UP] or keys[pygame.K_SPACE]

    if left and jump:
        return ACTION_LEFT_JUMP
    if right and jump:
        return ACTION_RIGHT_JUMP
    if left:
        return ACTION_LEFT
    if right:
        return ACTION_RIGHT
    if jump:
        return ACTION_JUMP
    
    return ACTION_NOTHING


def play_mario():
    # Mario enviroment
    config = InitConfig(level="1-1", character="luigi")
    env = SuperMarioBros2Env()

    obs, info = env.reset()

    # initialize pygame
    pygame.init()
    pygame.display.set_caption("Super Mario Bros 2 – Spielbar")

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