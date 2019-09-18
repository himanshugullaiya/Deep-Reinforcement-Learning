#%matplotlib inline
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T    

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

class DQN(nn.Module):
    def __init__(self, img_height, img_width):
        super().__init__()
            
        self.fc1 = nn.Linear(in_features=img_height*img_width*3, out_features=24)   
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)
    
    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size
    
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
    
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)

# reinforcement learning agent

class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        if  rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device) # explore      
        else:
            with torch.no_grad():
               return policy_net(state).argmax(dim=1).to(self.device) # exploit 

class CartPoleEnvManager:
    def __init__(self,device):
        self.device = device
        self.env = gym.make('CartPole-v0').unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False

    def reset(self):
        self.env.reset()
        self.current_screen = None
        
    def close(self):
        self.env.close()
    
    def render(self, mode='human'):
        return self.env.render(mode)
    
    def num_actions_available(self):
        return self.env.action_space.n
    
    def take_action(self, action):        
        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)
    
    def just_starting(self):
        return self.current_screen is None
    
    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1
    
    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]
    
    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]
    
    
    def crop_screen(self, screen):
        screen_height = screen.shape[1]
        top = int(screen_height * 0.4) # Strip off top and bottom
        bottom = int(screen_height * 0.8)
        screen = screen[:, top:bottom, :]
        return screen

    def transform_screen_data(self, screen):       
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
#        return screen
        resize = T.Compose([T.ToPILImage() ,T.Resize((40,90)) ,T.ToTensor()])
        return resize(screen).unsqueeze(0).to(self.device) # add a batch dimension (BCHW) 
    
    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1)) # PyTorch expects CHW
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

#**********************XxX************************#
'''
device = torch.device('cpu')
em = CartPoleEnvManager(device)

# Setting up instances
# Non-processed Screen
screen = em.render('rgb_array')
em.reset()
em.close()
plt.figure()
plt.imshow(screen)
plt.title('Non-processed screen example')
plt.show()

#Processed Scren Example
screen_2 = em.get_processed_screen()
em.reset()
em.close()
plt.figure()
plt.imshow(screen_2.squeeze(0).permute(1, 2, 0), interpolation='none')
plt.title('Processed screen example')
plt.show()

# Starting State
screen_3 = em.get_state()
plt.figure()
plt.imshow(screen_3.squeeze(0).permute(1, 2, 0), interpolation='none')
plt.title('Starting State example')
plt.show()

#.............Before action...........#
#screen_4 = em.get_state()
#plt.figure()
#plt.imshow(screen_3.squeeze(0).permute(1, 2, 0), interpolation='none')
#plt.title('Before Action')
#plt.show()

#Non_starting State
for _ in range(10):
    for x in range(5):
        if em.done:
            break
        em.take_action(torch.tensor(np.random.choice([0,1])))   # out of 1 or -1
    screen_4 = em.get_state()
    plt.figure()
    plt.imshow(screen_4.squeeze(0).permute(1, 2, 0), interpolation='none')
    plt.title('Non-Starting State example')
    plt.show()
    print(x)
em.reset()
em.close()
'''
#**********************XxX************************#
# utility Functions
def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()        
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)
    moving_avg = get_moving_average(values, moving_avg_period)
    plt.plot(moving_avg)
    plt.pause(0.001)
    print(f"Epsiode: {len(values)}'\n'{moving_avg_period} Episode Moving Avg: {moving_avg[-1]}")
    if is_ipython: display.clear_output(wait=True)
    
def get_moving_average(values, period):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

# tensor processing    
def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)
    
    return (t1, t2, t3, t4)

class QValues():
    device = torch.device('cuda')
    
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim = 1, index = actions.unsqueeze(-1))
    
    @staticmethod
    def get_next(target_net, next_states):                
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values
    
# Main Program
batch_size = 256
gamma = 0.999   # Discount Rate
max_exp = 1   # max exploration rate
min_exp = 0.01  # min_exp_rate
exp_decay = 0.001 
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 1000

device = torch.device('cuda')
em = CartPoleEnvManager(device)
strategy = EpsilonGreedyStrategy(max_exp, min_exp, exp_decay)
agent = Agent(strategy, em.num_actions_available(), device)
memory = ReplayMemory(memory_size)

policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
target_net.load_state_dict(policy_net.state_dict()) # set target's net's weights and baises as policy net's
target_net.eval() # tells compiler that this net will only  be used for inference & not training
optimizer = optim.Adam(params = policy_net.parameters(), lr = lr)  

episode_durations = []
for episode in range(num_episodes):
    em.reset()
    state = em.get_state()
    
    for timestep in count():
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state
        
        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)
            
            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if em.done:
            episode_durations.append(timestep)
            plot(episode_durations, 100)
            break
        
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())   
em.close()