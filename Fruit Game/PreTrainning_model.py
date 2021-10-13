import collections
import random

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000 # 버퍼 최대 크기
batch_size = 32


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            #print(a)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        input_size = 270
        self.conv1 = nn.Conv2d(9, 15, kernel_size = 3, stride = 1) #in_channels, out_channels
        self.conv2 = nn.Conv2d(15, 15, kernel_size= 3, stride = 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #2*2 maxpool

        self.fc1 = nn.Linear(input_size, input_size * 2 )
        self.fc2 = nn.Linear(input_size * 2, input_size * 2)
        self.fc3 = nn.Linear(input_size * 2, input_size)
        self.fc4 = nn.Linear(input_size, 170)  # 보드판 크기 170
        self.fc4_2 = nn.Linear(input_size, 170)

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = x.reshape(-1,10,17,9) #이거 알아보기
        x = x.permute(0,3,1,2) #이것도 알아보기

        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.reshape(-1, 270) #데이터 펼침
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x1 = self.fc4(x)
        x2 = self.fc4_2(x)
        return x1, x2    #grid를 받아서 grid를 배출..함

    def sample_action(self, obs, epsiloin): #observation 던져주면 epsilon에 따른 action을 던져줌
        out1, out2 = self.forward(obs)
        out1 = out1.reshape(10,17)
        out2 = out2.reshape(10,17)
        ##############Torch 를 Numpy 로 바꿔서 처리헀는데 여기서 오류가 날수도 있으니까 주의#############
        out1_xy = np.unravel_index(np.argmax(out1.detach().numpy(), axis=None), out1.shape)
        out2_xy = np.unravel_index(np.argmax(out2.detach().numpy(), axis=None), out2.shape)
        coordinate_p = [out1_xy , out2_xy]

        coin = random.random()
        if coin < epsiloin:
            select_coin = random.random()
            if(select_coin > 0.5):
                first = random.randint(1, 17)  # 여기 수정하기
                y1 = random.randint(1, 10)
                y2 = random.randint(1, 10)
                p1 = [first, y1]
                p2 = [first, y2]
                coordinate_p = [p1, p2]
            else:
                first = random.randint(1, 10)  # 여기 수정하기
                x1 = random.randint(1, 17)
                x2 = random.randint(1, 17)
                p1 = [x1, first]
                p2 = [x2, first]
                coordinate_p = [p1, p2]

        return coordinate_p   # 수정해야됨


multiplier = torch.zeros((4,2)) #.cuda 붙여서 gpu로 해도됨
multiplier[0,0] = 1.0
multiplier[1,0] = 17.0
multiplier[2,1] = 1.0
multiplier[3,1] = 17.0

def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        q_out = q(s) # q_out = ( torch.tensor(batch, 170), torch.tensor(batch,170))
        #print(a)
        #print(a.shape)
        q_out = torch.cat(q_out, dim = 1)

        a = torch.FloatTensor(a.view(-1,4).detach().numpy())

        a = torch.LongTensor(torch.matmul(a, multiplier).detach().numpy())
        #print(q_out.shape)
        q_a = q_out.gather(1, a)

        #max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        temp_0, temp_1 = q_target(s_prime)
        max_q_prime = torch.cat([temp_0.max(1)[0].unsqueeze(1),temp_1.max(1)[0].unsqueeze(1)], dim = 1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss



def IntegerEncoding(grid):
    token = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    word2index = {}

    for voca in token:
        if voca not in word2index.keys():
            word2index[voca] = len(word2index)

    new_array = np.zeros((10, 17, 9))  # 10 행 개수, 17 열 개수, 9개 채널 개수(one-hot)
    for i, x_value in enumerate(grid):
        for j, y_value in enumerate(x_value):
            one_hot_vector = [0] * 9   # 총 9개 채널
            index = word2index[y_value]
            if (index != 0):
                one_hot_vector[index - 1] = 1
            new_array[i, j] = one_hot_vector

    return new_array


# a = [(x1, y1) , (x2, y2)]
# @return s_prime, r, info
def game_step(a, env):
    p1 = a[0]
    p2 = a[1]
    next_state, reward = action(p1, p2, env)
    return next_state, float(reward)

from Game import *
from GameAlgorithm import *


def main():
    #env = gym.make('CartPole-v1') #env를 게임에 맞게 수정 후 사용

    q = Qnet() #input_size = 170(타일) * 9(채널)
    q_target = Qnet() # input_size = 170(타일) * 9(채널)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20  #20번시도마다 출력
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    episode_n = 0
    #200 * print_interval = 현재 총 4000번의 에피소드
    for i in range(200):
        best_score = -25
        best_state = np.zeros((10,17))

        for n_epi in range(100):
            epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
            s = (generate_fix_sum_random_vec(900, 170, tries=10)) # env 초기화 (새로운 배열 생성
            done = False
            tries = 0 # 10번 드래그해서 안되면 종료

            #프리트레이닝을 위한 알고리즘
            algorithm = ApplePy(s)
            coordinate_list = algorithm.run()
            list_size = len(coordinate_list)
            sequenceN = 0
            while not done:
                s_emb = IntegerEncoding(s)  # 원핫 인코딩된 게임 판
                #프리 트레이닝
                #--------------------랜덤 확률 -------------------
                select_coin = random.random()
                if (select_coin > 0.9):
                    a = q.sample_action(torch.from_numpy(s_emb).float(), epsilon)
                else:
                    a = coordinate_list[sequenceN]
                    sequenceN += 1
                #----------------여기까지 확률적으로 랜덤 ------------
                s_prime, r = game_step(a, s)

                s_prime_emb = IntegerEncoding(s_prime)
                if(r <= 0):   #아무것도 성공 못 하면
                    tries += 1
                done_mask = 0.0 if tries == 5 else 1.0
                #if(r >= -25): #어느정도 reward가 높지 않으면 버림
                memory.put((s_emb, a, r, s_prime_emb, done_mask))
                s = s_prime
                # if(r > 0):
                #      r = r + 100
                #      print("성공")
                score += r
                if tries == 5:
                    done = True
                if sequenceN == (list_size):
                    done = True

            if(best_score < score):
                best_score = score
                best_state = s

            loss = None
            if memory.size() > 2000:
                loss = train(q, q_target, memory, optimizer)

            # if n_epi % print_interval == 0 and n_epi != 0:
            if episode_n % print_interval == 0 and episode_n != 0:
                q_target.load_state_dict(q.state_dict())
                print("episode_th :{}, best score(40 try) : {:.2f}, n_buffer : {}, eps : {:.1f}%".format(
                    episode_n, best_score, memory.size(), epsilon * 100))
                if loss is not None :
                    print("loss : ", loss.item())
                print("zero : ", np.sum(best_state == 0))

            score = 0.0
            episode_n += 1

    path = './model/'
    torch.save(q, path + 'QModel.pt')
    torch.save(q.state_dict(), path + 'QModel_state_dict.pt')
    torch.save(q_target, path + 'targetModel.pt')
    torch.save(q_target.state_dict(), path + 'targetModel_state_dict.pt')

if __name__ == '__main__':
    main()

