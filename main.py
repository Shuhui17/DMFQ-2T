import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import copy
from tqdm import tqdm
import math
from scipy.stats import dirichlet
import time


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# 系统参数配置
class SystemConfig:
    def __init__(self):
        # 系统参数
        self.N = 50  # 用户数量
        self.T = 1000  # 时间槽数量
        self.time_slot_length = 0.1 

        # 计算参数
        self.F_e = 20e9  # MEC服务器计算能力(cycles/s)
        self.user_capacity_range = [0.1e9, 1e9]  # 用户设备计算能力范围(cycles/s)
        self.k = 1e-27  # 能量系数

        # 通信参数
        self.W = 20e6  # 无线信道带宽(Hz)
        self.user_transmission_power_range = [0.1, 0.5]  # 用户传输功率范围(W)
        self.noise_power = 2e-16  # 噪声功率
        self.communication_range = 300  # 通信范围(m)
        self.area_size = 500  # 区域大小(m)

        # 任务参数
        self.data_size_range = [2e5, 4e5]  # 任务数据大小范围(bits)
        self.computation_range = [6e8, 9e8]  # 任务计算量范围(cycles)

        # 学习参数
        self.hidden_dim = 128
        self.batch_size = 128
        self.learning_rate = 0.0001
        self.gamma = 0.95  # 折扣因子
        self.target_network_tau = 0.01 
        self.memory_size = 10000
        self.num_episodes = 2000
        self.max_steps = 50
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995

        # 平均场参数
        self.dirichlet_samples = 100
        self.dirichlet_prior = 1.0
        self.boltzmann_beta = 1.0

        # MADDPG参数
        self.critic_lr = 0.001
        self.actor_lr = 0.0001

        # POMF-AC参数
        self.pomf_ac_actor_lr = 0.0001
        self.pomf_ac_critic_lr = 0.001


config = SystemConfig()


# 用户类型定义
class UserType:
    LATENCY_SENSITIVE = 0  # 延迟敏感型
    ENERGY_SENSITIVE = 1  # 能耗敏感型


# 任务类
class Task:
    def __init__(self, user_id, time_slot):
        self.user_id = user_id
        self.time_slot = time_slot
        self.data_size = np.random.uniform(config.data_size_range[0], config.data_size_range[1])  # D_i(t)
        self.computation_requirement = np.random.uniform(config.computation_range[0],
                                                         config.computation_range[1])  # C_i(t)
        self.arrival_time = time_slot * config.time_slot_length 
        self.start_time = None
        self.completion_time = None


# 系统环境
class MECEnvironment:
    def __init__(self, num_users, user_type_ratio=0.5):
        self.num_users = num_users
        self.user_type_ratio = user_type_ratio
        self.users = []
        self.time_step = 0
        self.current_time = 0

        # 初始化用户
        self.reset()

    def reset(self):
        self.users = []
        for i in range(self.num_users):
            # 随机分配用户类型
            user_type = UserType.LATENCY_SENSITIVE if random.random() < self.user_type_ratio else UserType.ENERGY_SENSITIVE

            # 随机用户参数
            f_i_l = np.random.uniform(config.user_capacity_range[0], config.user_capacity_range[1])  # 本地计算能力
            p_i = np.random.uniform(config.user_transmission_power_range[0],
                                    config.user_transmission_power_range[1])  # 传输功率
            h_i = np.random.uniform(0.5, 1.0)  # 信道增益

            # 随机位置
            x = random.uniform(0, config.area_size)
            y = random.uniform(0, config.area_size)

            user = {
                'id': i,
                'type': user_type,
                'f_i_l': f_i_l,
                'p_i': p_i,
                'h_i': h_i,
                'location': (x, y),  # 用户位置
                'velocity': (0, 0),  # 用户移动速度
                'local_queue': deque(),  # 本地执行队列
                'transmission_queue': deque(),  # 传输队列
                'local_waiting_time': 0.0,  # 本地等待时间
                'transmission_waiting_time': 0.0,  # 传输等待时间
                'mec_waiting_time': 0.0,  # MEC执行等待时间
                'neighbors': [],
                'same_type_neighbors': [],
                'diff_type_neighbors': [],
                'action_history': [],
                'is_transmitting': False,  # 初始化为False
                'current_task': None,
                'prev_local_waiting_time': 0.0,  # 上一时隙的本地等待时间
                'prev_transmission_waiting_time': 0.0,  # 上一时隙的传输等待时间
                'prev_mec_waiting_time': 0.0,  # 上一时隙的MEC等待时间
                'prev_action': None,  # 上一时隙的动作
                'prev_task': None,  # 上一时隙的任务
                'prev_transmission_rate': 0.0,  # 上一时隙的传输速率
                'prev_active_users': 0  # 上一时隙的活跃用户数
            }
            self.users.append(user)

       
        self.dt_server = {
            'user_twins': copy.deepcopy(self.users),
            'cumulative_action_counts': defaultdict(lambda: {
                'same_type': np.array([config.dirichlet_prior, config.dirichlet_prior]),
                'diff_type': np.array([config.dirichlet_prior, config.dirichlet_prior])
            })
        }

        self.time_step = 0
        self.current_time = 0

        # 生成初始任务
        for user in self.users:
            user['current_task'] = Task(user['id'], self.time_step)

        # 更新邻居信息
        self.update_neighbors()

        # 更新本地观察
        self.update_local_observations()

        return self.get_local_observations()

    def update_neighbors(self):
        # 根据距离更新邻居信息
        for i, user in enumerate(self.users):
            neighbors = []
            same_type_neighbors = []
            diff_type_neighbors = []

            for j, other_user in enumerate(self.users):
                if i != j:
                    # 计算欧几里得距离
                    dist = math.sqrt((user['location'][0] - other_user['location'][0]) ** 2 +
                                     (user['location'][1] - other_user['location'][1]) ** 2)

                    # 如果距离小于通信范围，则认为是邻居
                    if dist <= config.communication_range:
                        neighbors.append(j)
                        if other_user['type'] == user['type']:
                            same_type_neighbors.append(j)
                        else:
                            diff_type_neighbors.append(j)

            user['neighbors'] = neighbors
            user['same_type_neighbors'] = same_type_neighbors
            user['diff_type_neighbors'] = diff_type_neighbors

    def update_user_locations(self):
        # 更新用户位置（随机移动）
        for user in self.users:
            x, y = user['location']
            vx, vy = user['velocity']

            # 更新位置
            new_x = x + vx * config.time_slot_length  # 修改为time_slot_length
            new_y = y + vy * config.time_slot_length  # 修改为time_slot_length

            # 边界检查
            if new_x < 0 or new_x > config.area_size:
                vx = -vx 
                new_x = max(0, min(config.area_size, new_x))
            if new_y < 0 or new_y > config.area_size:
                vy = -vy 
                new_y = max(0, min(config.area_size, new_y))

            user['location'] = (new_x, new_y)
            user['velocity'] = (vx, vy)

    def update_local_observations(self):
        
        for i, user in enumerate(self.users):
            # 自身状态：D_i(t), C_i(t), f_i^l, p_i, h_i
            observation = [
                user['current_task'].data_size,
                user['current_task'].computation_requirement,
                user['f_i_l'],
                user['p_i'],
                user['h_i'],
                user['local_waiting_time'],
                user['transmission_waiting_time'],
                user['mec_waiting_time'],
                user['type'],
                1 if user['is_transmitting'] else 0
            ]

           
            same_type_actions = []
            diff_type_actions = []
            same_type_data_sizes = []
            same_type_comp_reqs = []
            diff_type_data_sizes = []
            diff_type_comp_reqs = []

            for neighbor_id in user['neighbors']:
                neighbor_twin = self.dt_server['user_twins'][neighbor_id]
                if len(neighbor_twin['action_history']) > 0:
                    if neighbor_twin['type'] == user['type']:
                        same_type_actions.append(neighbor_twin['action_history'][-1])
                        same_type_data_sizes.append(neighbor_twin['current_task'].data_size)
                        same_type_comp_reqs.append(neighbor_twin['current_task'].computation_requirement)
                    else:
                        diff_type_actions.append(neighbor_twin['action_history'][-1])
                        diff_type_data_sizes.append(neighbor_twin['current_task'].data_size)
                        diff_type_comp_reqs.append(neighbor_twin['current_task'].computation_requirement)

      
            observation.append(np.mean(same_type_actions) if same_type_actions else 0.5)
            observation.append(np.mean(same_type_data_sizes) if same_type_data_sizes else 0)
            observation.append(np.mean(same_type_comp_reqs) if same_type_comp_reqs else 0)
            observation.append(len(same_type_actions))

            observation.append(np.mean(diff_type_actions) if diff_type_actions else 0.5)
            observation.append(np.mean(diff_type_data_sizes) if diff_type_data_sizes else 0)
            observation.append(np.mean(diff_type_comp_reqs) if diff_type_comp_reqs else 0)
            observation.append(len(diff_type_actions))

            user['local_observation'] = np.array(observation, dtype=np.float32)

    def get_local_observations(self):
        observations = []
        for user in self.users:
            observations.append(user['local_observation'])
        return np.array(observations)

    def get_global_state(self):
        # 获取全局状态（用于完全可观测算法）
        global_state = []
        for user in self.users:
            global_state.extend(user['local_observation'])
        return np.array(global_state)

    def calculate_transmission_rate(self, user):
        # 计算传输速率 
        # 使用NOMA技术，考虑其他传输用户的干扰
        interference = 0
        transmitting_users = [u for u in self.users if u['is_transmitting']]

        # 计算干扰项
        for other_user in transmitting_users:
            if other_user['id'] != user['id']:
                interference += other_user['p_i'] * other_user['h_i']

        sinr = (user['p_i'] * user['h_i']) / (config.noise_power + interference)
        return config.W * math.log2(1 + sinr)  # bps

    def compute_local_execution_time(self, user, task):
        # 计算本地执行时间
        return task.computation_requirement / user['f_i_l']  # T_{i,exec}^l(t)

    def compute_local_waiting_time(self, user, prev_task, prev_action):
        # 计算本地等待时间
        if self.time_step == 0:
            return 0.0
        else:
            # 计算上一时隙的本地执行时间
            prev_exec_time = 0.0
            if prev_action == 0 and prev_task is not None:  # 上一时隙本地执行
                prev_exec_time = prev_task.computation_requirement / user['f_i_l']

            # 计算本地等待时间
            local_waiting_time = max(0, user[
                'prev_local_waiting_time'] + prev_exec_time - config.time_slot_length) 
            return local_waiting_time

    def compute_local_energy(self, user, task):
        # 计算本地能耗
        return config.k * (user['f_i_l'] ** 2) * task.computation_requirement

    def compute_transmission_time(self, user, task):
        # 计算传输时间
        transmission_rate = self.calculate_transmission_rate(user)
        return task.data_size / transmission_rate  # T_{i,trans}^{off}(t)

    def compute_transmission_waiting_time(self, user, prev_task, prev_action, prev_transmission_rate):
        # 计算传输等待时间
        if self.time_step == 0:
            return 0.0
        else:
            # 计算上一时隙的传输时间
            prev_transmission_time = 0.0
            if prev_action == 1 and prev_task is not None:  # 上一时隙卸载执行
                prev_transmission_time = prev_task.data_size / prev_transmission_rate

            # 计算传输等待时间
            current_transmission_rate = self.calculate_transmission_rate(user)
            transmission_waiting_time = max(0, user['prev_transmission_waiting_time'] +
                                            prev_transmission_time - config.time_slot_length)  # 修改为time_slot_length

            # 调整传输等待时间（考虑传输速率变化）
            if current_transmission_rate != 0:
                transmission_waiting_time *= (prev_transmission_rate / current_transmission_rate)

            return transmission_waiting_time

    def compute_mec_execution_time(self, user, task):
        # 计算MEC执行时间 - 公式(9)
        # 根据论文公式(11)计算活跃用户数
        active_users = self.get_active_users_count()
        if active_users == 0:
            active_users = 1
        return (task.computation_requirement / config.F_e) * active_users  # T_{i,exe}^{off}(t)

    def compute_mec_waiting_time(self, user, prev_task, prev_action, prev_active_users):
        # 计算MEC等待时间 - 公式(10)
        if self.time_step == 0:
            return 0.0
        else:
            # 计算上一时隙的MEC执行时间
            prev_mec_exec_time = 0.0
            if prev_action == 1 and prev_task is not None:  # 上一时隙卸载执行
                prev_mec_exec_time = (prev_task.computation_requirement / config.F_e) * prev_active_users

            # 计算MEC等待时间
            # 根据论文公式(11)计算当前活跃用户数
            current_active_users = self.get_active_users_count()
            mec_waiting_time = max(0, user[
                'prev_mec_waiting_time'] + prev_mec_exec_time - config.time_slot_length)  # 修改为time_slot_length

            # 调整MEC等待时间（考虑活跃用户数变化）- 根据论文公式(10)
            if prev_active_users != 0:
                mec_waiting_time *= (current_active_users / prev_active_users)  # 修正：使用current/prev

            return mec_waiting_time

    def get_active_users_count(self):
        # 根据论文公式(11)计算活跃用户数
        # b'_k(t) = 0 当 a_k(t)=0 且 T^{off}_{k,exe_wait}(t)=0
        # b'_k(t) = 1 其他情况
        active_count = 0
        for user in self.users:
            # 使用当前时刻的is_transmitting属性来判断用户是否在传输
            # 考虑传输等待队列不为空的情况
            if user['is_transmitting']:  # 当前时隙选择卸载或传输队列不为空
                active_count += 1
            elif user['mec_waiting_time'] > 0:  # 有MEC等待时间
                active_count += 1
        return active_count

    def compute_offloading_energy(self, user, task):
        # 计算卸载能耗 - 公式(13)
        transmission_time = self.compute_transmission_time(user, task)
        return user['p_i'] * transmission_time

    def update_queues(self):
        # 更新所有队列状态 - 修改为无论当前决策如何，都处理队列中的任务
        transmitting_users = []

        # 处理传输队列 - 无论当前决策如何，只要传输队列不为空就处理
        for user in self.users:
            # 设置传输状态 - 只要传输队列不为空，用户就在传输
            user['is_transmitting'] = len(user['transmission_queue']) > 0

            if user['is_transmitting']:
                transmitting_users.append(user['id'])

        for user in self.users:

            if user['is_transmitting']:
                # 处理传输队列中的所有任务（在当前时隙内尽可能多地传输）
                remaining_time = config.time_slot_length  # 修改为time_slot_length
                while user['transmission_queue'] and remaining_time > 0:
                    task = user['transmission_queue'][0]
                    transmission_rate = self.calculate_transmission_rate(user)
                    transmission_time = task.data_size / transmission_rate

                    if transmission_time <= remaining_time:
                        # 任务传输完成
                        user['transmission_queue'].popleft()
                        remaining_time -= transmission_time
                    else:
                        # 任务只传输了一部分
                        transmitted_data = transmission_rate * remaining_time
                        task.data_size -= transmitted_data
                        remaining_time = 0

        # 处理本地队列 - 无论当前决策如何，只要本地队列不为空就处理
        for user in self.users:
            # 处理本地队列中的所有任务（在当前时隙内尽可能多地执行）
            remaining_time = config.time_slot_length  # 修改为time_slot_length
            while user['local_queue'] and remaining_time > 0:
                task = user['local_queue'][0]
                exec_time = task.computation_requirement / user['f_i_l']

                if exec_time <= remaining_time:
                    # 任务执行完成
                    user['local_queue'].popleft()
                    remaining_time -= exec_time
                else:
                    # 任务只执行了一部分
                    executed_cycles = user['f_i_l'] * remaining_time
                    task.computation_requirement -= executed_cycles
                    remaining_time = 0

    def step(self, actions):
        rewards = []

        # 更新用户位置
        self.update_user_locations()

        # 第一步：根据当前决策和传输队列状态更新is_transmitting属性
        for i, user in enumerate(self.users):
            action = actions[i]

            # 记录动作历史
            user['action_history'].append(action)

            # 更新is_transmitting属性 - 考虑当前决策和传输等待队列
            # 如果当前决策是卸载执行，或者传输队列不为空，则设置为True
            user['is_transmitting'] = (action == 1) or (len(user['transmission_queue']) > 0)

        # 计算当前时隙的活跃用户数
        current_active_users = self.get_active_users_count()

        # 第二步：计算传输速率（基于更新后的is_transmitting状态）
        for i, user in enumerate(self.users):
            # 计算传输速率
            transmission_rate = self.calculate_transmission_rate(user)

            # 保存传输速率用于后续计算
            user['current_transmission_rate'] = transmission_rate

        # 第三步：处理每个用户的决策和任务
        for i, user in enumerate(self.users):
            action = actions[i]
            task = user['current_task']
            transmission_rate = user['current_transmission_rate']

            # 根据论文公式计算等待时间
            local_waiting_time = self.compute_local_waiting_time(
                user, user['prev_task'], user['prev_action'])

            transmission_waiting_time = self.compute_transmission_waiting_time(
                user, user['prev_task'], user['prev_action'], user['prev_transmission_rate'])

            mec_waiting_time = self.compute_mec_waiting_time(
                user, user['prev_task'], user['prev_action'], user['prev_active_users'])

            # 更新用户等待时间
            user['local_waiting_time'] = local_waiting_time
            user['transmission_waiting_time'] = transmission_waiting_time
            user['mec_waiting_time'] = mec_waiting_time

            if action == 0:  # 本地执行
                # 计算本地执行时间
                local_exec_time = self.compute_local_execution_time(user, task)

                # 计算总延迟
                latency = local_exec_time + local_waiting_time

                # 计算能耗
                energy = self.compute_local_energy(user, task)

                # 将任务加入本地队列
                user['local_queue'].append(task)
            else:  # 卸载执行
                # 计算传输时间
                transmission_time = self.compute_transmission_time(user, task)

                # 计算MEC执行时间
                mec_exec_time = self.compute_mec_execution_time(user, task)

                # 计算总延迟
                latency = (transmission_time + transmission_waiting_time +
                           mec_exec_time + mec_waiting_time)

                # 计算能耗
                energy = self.compute_offloading_energy(user, task)

                # 将任务加入传输队列
                user['transmission_queue'].append(task)

            # 根据用户类型计算奖励
            if user['type'] == UserType.LATENCY_SENSITIVE:
                reward = -latency  # 最小化延迟
            else:
                reward = -energy  # 最小化能耗

            rewards.append(reward)

            # 保存当前时隙信息用于下一时隙计算
            user['prev_action'] = action
            user['prev_task'] = task
            user['prev_transmission_rate'] = transmission_rate
            user['prev_local_waiting_time'] = local_waiting_time
            user['prev_transmission_waiting_time'] = transmission_waiting_time
            user['prev_mec_waiting_time'] = mec_waiting_time

        # 第四步：更新队列状态
        self.update_queues()

        # 更新时间
        self.time_step += 1
        self.current_time += config.time_slot_length  # 修改为time_slot_length

        # 生成新任务
        for user in self.users:
            user['current_task'] = Task(user['id'], self.time_step)
            user['prev_active_users'] = current_active_users

        # 更新邻居信息（因为用户位置发生了变化）
        self.update_neighbors()

       
        self.update_digital_twin()

        # 更新累计动作计数
        self.update_cumulative_action_counts()

       
        self.update_local_observations()

        next_observations = self.get_local_observations()
        done = self.time_step >= config.max_steps

        return next_observations, rewards, done

    def update_digital_twin(self):
        self.dt_server['user_twins'] = copy.deepcopy(self.users)

    def update_cumulative_action_counts(self):
        # 更新累计动作计数
        for i, user in enumerate(self.users):
            # 同类型邻居动作计数
            same_type_counts = np.array([config.dirichlet_prior, config.dirichlet_prior])
            for neighbor_id in user['same_type_neighbors']:
                neighbor = self.users[neighbor_id]
                if neighbor['action_history']:
                    action = neighbor['action_history'][-1]
                    same_type_counts[action] += 1

            # 不同类型邻居动作计数
            diff_type_counts = np.array([config.dirichlet_prior, config.dirichlet_prior])
            for neighbor_id in user['diff_type_neighbors']:
                neighbor = self.users[neighbor_id]
                if neighbor['action_history']:
                    action = neighbor['action_history'][-1]
                    diff_type_counts[action] += 1

            self.dt_server['cumulative_action_counts'][i]['same_type'] = same_type_counts
            self.dt_server['cumulative_action_counts'][i]['diff_type'] = diff_type_counts

    def get_mean_field_actions(self, user_id):
        # 使用Dirichlet分布估计平均场动作
        user = self.users[user_id]
        same_type_params = self.dt_server['cumulative_action_counts'][user_id]['same_type']
        diff_type_params = self.dt_server['cumulative_action_counts'][user_id]['diff_type']

        # 采样得到平均场动作
        same_type_mean = 0.5
        diff_type_mean = 0.5

        if np.sum(same_type_params) > 0:
            same_type_samples = dirichlet.rvs(same_type_params, size=config.dirichlet_samples, random_state=SEED)
            same_type_mean = np.mean(same_type_samples[:, 1])  # 动作1的概率

        if np.sum(diff_type_params) > 0:
            diff_type_samples = dirichlet.rvs(diff_type_params, size=config.dirichlet_samples, random_state=SEED)
            diff_type_mean = np.mean(diff_type_samples[:, 1])  # 动作1的概率

        return same_type_mean, diff_type_mean

    def get_single_mean_field_actions(self, user_id):
        # 只考虑一个平均场，不考虑用户类型
        user = self.users[user_id]
        all_neighbor_params = np.array([config.dirichlet_prior, config.dirichlet_prior])

        for neighbor_id in user['neighbors']:
            neighbor = self.users[neighbor_id]
            if neighbor['action_history']:
                action = neighbor['action_history'][-1]
                all_neighbor_params[action] += 1

        # 采样得到平均场动作
        single_mean = 0.5
        if np.sum(all_neighbor_params) > 0:
            all_samples = dirichlet.rvs(all_neighbor_params, size=config.dirichlet_samples, random_state=SEED)
            single_mean = np.mean(all_samples[:, 1])  # 动作1的概率

        return single_mean


# 部分可观测平均场Q网络
class POMFQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(POMFQNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        
        self.state_dense = nn.Linear(state_dim, 64)  # 将状态输入映射到64维

        # 平均场输入层
        self.mf_dense1 = nn.Linear(2, 64)  # 输入为2维（同类型和不同类型平均场动作）
        self.mf_dense2 = nn.Linear(64, 32)

        # 中间层
        self.dense2 = nn.Linear(64 + 32, 128)  # 状态输出64维 + 平均场输出32维
        self.dense_out = nn.Linear(128, 64)

        # 输出层
        self.q_output = nn.Linear(64, action_dim)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, state, same_type_mean, diff_type_mean):
        # 处理状态输入
        h_state = self.relu(self.state_dense(state))

        
        # 将同类型和不同类型平均场动作拼接为2维向量
        mf_input = torch.stack([same_type_mean, diff_type_mean], dim=-1)
        prob_emb = self.relu(self.mf_dense1(mf_input))
        h_act_prob = self.relu(self.mf_dense2(prob_emb))

        # 拼接状态和平均场特征
        concat_layer = torch.cat([h_state, h_act_prob], dim=-1)

        # 通过中间层
        dense2_out = self.relu(self.dense2(concat_layer))
        out = self.relu(self.dense_out(dense2_out))

        # 输出Q值
        q_values = self.q_output(out)

        return q_values


# 独立Q学习网络 (IQL)
class IQLNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(IQLNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)


# MADDPG Actor网络
class MADDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MADDPGActor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)


# MADDPG Critic网络
class MADDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super(MADDPGCritic, self).__init__()
        # 输入是所有智能体的状态和动作的拼接
        # 每个智能体：state_dim维状态 + action_dim维动作
        # 总共：num_agents * (state_dim + action_dim)
        input_dim = num_agents * (state_dim + action_dim)

        self.network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )

    def forward(self, states, actions):
        # 将状态和动作拼接
        x = torch.cat([states, actions], dim=-1)
        return self.network(x)


# 单平均场Q网络 (不考虑用户类型)
class SingleMeanFieldQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SingleMeanFieldQNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_dense = nn.Linear(state_dim, 64)
        self.mf_dense1 = nn.Linear(1, 64)  # 输入为1维（单一平均场动作）
        self.mf_dense2 = nn.Linear(64, 32)

        self.dense2 = nn.Linear(64 + 32, 128)
        self.dense_out = nn.Linear(128, 64)
        self.q_output = nn.Linear(64, action_dim)

        self.relu = nn.ReLU()

    def forward(self, state, single_mean):
        h_state = self.relu(self.state_dense(state))

        mf_input = single_mean.unsqueeze(-1)
        prob_emb = self.relu(self.mf_dense1(mf_input))
        h_act_prob = self.relu(self.mf_dense2(prob_emb))

        concat_layer = torch.cat([h_state, h_act_prob], dim=-1)

        dense2_out = self.relu(self.dense2(concat_layer))
        out = self.relu(self.dense_out(dense2_out))

        q_values = self.q_output(out)

        return q_values


# 完全可观测平均场Q网络
class FullObsMeanFieldQNetwork(nn.Module):
    def __init__(self, global_state_dim, action_dim):
        super(FullObsMeanFieldQNetwork, self).__init__()
        self.global_state_dim = global_state_dim
        self.action_dim = action_dim

        self.state_dense = nn.Linear(global_state_dim, 128)
        self.mf_dense1 = nn.Linear(2, 64)
        self.mf_dense2 = nn.Linear(64, 32)

        self.dense2 = nn.Linear(128 + 32, 256)
        self.dense_out = nn.Linear(256, 128)
        self.q_output = nn.Linear(128, action_dim)

        self.relu = nn.ReLU()

    def forward(self, global_state, same_type_mean, diff_type_mean):
        h_state = self.relu(self.state_dense(global_state))

        mf_input = torch.stack([same_type_mean, diff_type_mean], dim=-1)
        prob_emb = self.relu(self.mf_dense1(mf_input))
        h_act_prob = self.relu(self.mf_dense2(prob_emb))

        concat_layer = torch.cat([h_state, h_act_prob], dim=-1)

        dense2_out = self.relu(self.dense2(concat_layer))
        out = self.relu(self.dense_out(dense2_out))

        q_values = self.q_output(out)

        return q_values


# POMF Actor-Critic Actor网络
class POMFACActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(POMFACActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 状态输入层
        self.state_dense = nn.Linear(state_dim, 64)

        # 平均场输入层
        self.mf_dense1 = nn.Linear(2, 64)
        self.mf_dense2 = nn.Linear(64, 32)

        # 中间层
        self.dense2 = nn.Linear(64 + 32, 128)
        self.dense_out = nn.Linear(128, 64)

        # 输出层 - 策略输出
        self.policy_output = nn.Linear(64, action_dim)

        # 激活函数
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state, same_type_mean, diff_type_mean):
        # 处理状态输入
        h_state = self.relu(self.state_dense(state))

        # 处理平均场输入
        mf_input = torch.stack([same_type_mean, diff_type_mean], dim=-1)
        prob_emb = self.relu(self.mf_dense1(mf_input))
        h_act_prob = self.relu(self.mf_dense2(prob_emb))

        # 拼接状态和平均场特征
        concat_layer = torch.cat([h_state, h_act_prob], dim=-1)

        # 通过中间层
        dense2_out = self.relu(self.dense2(concat_layer))
        out = self.relu(self.dense_out(dense2_out))

        # 输出策略概率
        policy = self.softmax(self.policy_output(out))

        return policy


# POMF Actor-Critic Critic网络
class POMFACCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(POMFACCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 状态输入层
        self.state_dense = nn.Linear(state_dim, 64)

        # 平均场输入层
        self.mf_dense1 = nn.Linear(2, 64)
        self.mf_dense2 = nn.Linear(64, 32)

        # 动作输入层
        self.action_dense = nn.Linear(action_dim, 32)

        # 中间层
        self.dense2 = nn.Linear(64 + 32 + 32, 128)
        self.dense_out = nn.Linear(128, 64)

        # 输出层 - Q值输出
        self.q_output = nn.Linear(64, 1)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, state, action, same_type_mean, diff_type_mean):
        # 处理状态输入
        h_state = self.relu(self.state_dense(state))

        # 处理平均场输入
        mf_input = torch.stack([same_type_mean, diff_type_mean], dim=-1)
        prob_emb = self.relu(self.mf_dense1(mf_input))
        h_act_prob = self.relu(self.mf_dense2(prob_emb))

        # 处理动作输入
        h_action = self.relu(self.action_dense(action))

        # 拼接状态、平均场和动作特征
        concat_layer = torch.cat([h_state, h_act_prob, h_action], dim=-1)

        # 通过中间层
        dense2_out = self.relu(self.dense2(concat_layer))
        out = self.relu(self.dense_out(dense2_out))

        # 输出Q值
        q_value = self.q_output(out)

        return q_value


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, same_type_mean, diff_type_mean):
        self.buffer.append((state, action, reward, next_state, done, same_type_mean, diff_type_mean))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, same_type_mean, diff_type_mean = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), np.array(next_state),
                np.array(done), np.array(same_type_mean), np.array(diff_type_mean))

    def __len__(self):
        return len(self.buffer)


# MADDPG经验回放缓冲区
class MADDPGReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, states, actions, rewards, next_states, dones):
        self.buffer.append((states, actions, rewards, next_states, dones))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 确保返回numpy数组
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# 部分可观测平均场Q学习算法
class POMFQLearningAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Q网络和目标网络
        self.q_net = POMFQNetwork(state_dim, action_dim)
        self.target_q_net = POMFQNetwork(state_dim, action_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.learning_rate)
        self.memory = ReplayBuffer(config.memory_size)

        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay

      
        self.temperature = 0.1

    def get_action(self, state, same_type_mean, diff_type_mean, exploration=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        same_type_tensor = torch.FloatTensor([same_type_mean])
        diff_type_tensor = torch.FloatTensor([diff_type_mean])

        with torch.no_grad():
            q_values = self.q_net(state_tensor, same_type_tensor, diff_type_tensor)

            # 使用softmax和温度参数选择动作
            probabilities = torch.softmax(q_values / self.temperature, dim=-1)
            action = torch.argmax(probabilities, dim=-1).item()

        return action

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return 0

        # 采样批次数据
        states, actions, rewards, next_states, dones, same_type_means, diff_type_means = self.memory.sample(batch_size)

        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        same_type_means = torch.FloatTensor(same_type_means)
        diff_type_means = torch.FloatTensor(diff_type_means)

        # 计算当前Q值
        current_q_values = self.q_net(states, same_type_means, diff_type_means)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states, same_type_means, diff_type_means)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + config.gamma * max_next_q_values * (~dones)

        # 计算损失
        loss = torch.mean((target_q_values - current_q_values) ** 2)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 衰减epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def soft_update(self):
        # 软更新目标网络
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(target_param.data * (
                    1.0 - config.target_network_tau) + param.data * config.target_network_tau)  # 修改为target_network_tau

    def save(self, path):
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_q_net_state_dict': self.target_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# 对比算法
class RandomAgent:
    def get_action(self, state, exploration=True):
        return random.randint(0, 1)


class LocalExecutionAgent:
    def get_action(self, state, exploration=True):
        return 0


class OffloadingAgent:
    def get_action(self, state, exploration=True):
        return 1


# 独立Q学习算法 (IQL)
class IQLAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Q网络和目标网络
        self.q_net = IQLNetwork(state_dim, action_dim)
        self.target_q_net = IQLNetwork(state_dim, action_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.learning_rate)
        self.memory = deque(maxlen=config.memory_size)

        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay

    def get_action(self, state, same_type_mean=None, diff_type_mean=None, exploration=True):
        if exploration and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
                action = torch.argmax(q_values).item()
            return action

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return 0

        # 采样批次数据
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        # 计算当前Q值
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(1)[0]
            target_q_values = rewards + config.gamma * next_q_values * (~dones)

        # 计算损失
        loss = F.mse_loss(current_q_values, target_q_values)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 衰减epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def soft_update(self):
        # 软更新目标网络
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - config.target_network_tau) + param.data * config.target_network_tau)

    def push(self, state, action, reward, next_state, done, same_type_mean=None, diff_type_mean=None):
        self.memory.append((state, action, reward, next_state, done))


# MADDPG算法
class MADDPGAgent:
    def __init__(self, state_dim, action_dim, num_agents, agent_id):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.agent_id = agent_id

        # Actor网络
        self.actor = MADDPGActor(state_dim, action_dim)
        self.target_actor = MADDPGActor(state_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)

        # Critic网络
        self.critic = MADDPGCritic(state_dim, action_dim, num_agents)
        self.target_critic = MADDPGCritic(state_dim, action_dim, num_agents)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.memory = MADDPGReplayBuffer(config.memory_size)
        self.noise = OUNoise(action_dim)

    def get_action(self, state, same_type_mean=None, diff_type_mean=None, exploration=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state_tensor)

        if exploration:
            # 添加探索噪声
            noise = self.noise.sample()
            action_probs_np = action_probs.detach().numpy()[0] + noise

            # 确保概率值在有效范围内
            action_probs_np = np.clip(action_probs_np, 1e-8, 1.0)  # 避免0和负数

            # 归一化，确保概率和为1
            prob_sum = np.sum(action_probs_np)
            if prob_sum <= 0 or np.isnan(prob_sum):
                # 如果概率和无效，使用均匀分布
                action_probs_np = np.ones(self.action_dim) / self.action_dim
            else:
                action_probs_np = action_probs_np / prob_sum

            # 再次检查NaN值
            if np.any(np.isnan(action_probs_np)):
                action_probs_np = np.ones(self.action_dim) / self.action_dim

            action = np.random.choice(self.action_dim, p=action_probs_np)
        else:
            action = torch.argmax(action_probs).item()

        return action

    def update(self, batch_size, agents):
        if len(self.memory) < batch_size:
            return 0, 0

        try:
            # 采样批次数据
            states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

            # 检查数据有效性
            if np.any(np.isnan(states)) or np.any(np.isnan(actions)) or np.any(np.isnan(rewards)) or np.any(
                    np.isnan(next_states)):
                print("Warning: NaN values detected in batch data")
                return 0, 0

            # 转换为张量
            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.BoolTensor(dones)

            agent_id = self.agent_id
            batch_size_t = states.shape[0]

            # 更新Critic
            with torch.no_grad():
                # 计算目标动作
                target_actions = []
                for i in range(self.num_agents):
                    # 提取单个智能体的下一状态
                    agent_next_state = next_states[:, i * self.state_dim:(i + 1) * self.state_dim]
                    target_action_probs = agents[i].target_actor(agent_next_state)
                    # 检查目标动作概率
                    if torch.any(torch.isnan(target_action_probs)):
                        target_action_probs = torch.ones_like(target_action_probs) / self.action_dim
                    target_actions.append(target_action_probs)

                # 拼接所有智能体的目标动作
                target_actions = torch.cat(target_actions, dim=1)

                # 计算目标Q值
                target_q_values = self.target_critic(next_states, target_actions).squeeze()
                target_q_values = rewards[:, agent_id] + config.gamma * target_q_values * (~dones[:, agent_id])

            # 计算当前Q值
            current_q_values = self.critic(states, actions).squeeze()

            # Critic损失
            critic_loss = F.mse_loss(current_q_values, target_q_values)

            # 优化Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()

            # 更新Actor
            # 计算当前动作
            current_actions = []
            for i in range(self.num_agents):
                if i == agent_id:
                    # 当前智能体使用actor网络计算新动作
                    agent_state = states[:, i * self.state_dim:(i + 1) * self.state_dim]
                    action_probs = self.actor(agent_state)
                    # 检查动作概率
                    if torch.any(torch.isnan(action_probs)):
                        action_probs = torch.ones_like(action_probs) / self.action_dim
                    current_actions.append(action_probs)
                else:
                    # 其他智能体的动作保持不动，但需要确保维度正确
                  
                    agent_actions = actions[:, i * self.action_dim:(i + 1) * self.action_dim]
                    current_actions.append(agent_actions)

            # 拼接所有智能体的动作
            current_actions = torch.cat(current_actions, dim=1)

            # Actor损失
            actor_loss = -self.critic(states, current_actions).mean()

            # 优化Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # 软更新目标网络
            self.soft_update()

            return actor_loss.item(), critic_loss.item()

        except Exception as e:
            print(f"Error in MADDPG update: {e}")
            import traceback
            traceback.print_exc()
            return 0, 0

    def soft_update(self):
        # 软更新目标网络
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - config.target_network_tau) + param.data * config.target_network_tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - config.target_network_tau) + param.data * config.target_network_tau)

    def push(self, states, actions, rewards, next_states, dones):
        self.memory.push(states, actions, rewards, next_states, dones)


class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.1):  # 减小sigma值
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx

        # 限制噪声范围
        self.state = np.clip(self.state, -1.0, 1.0)
        return self.state


# 单平均场Q学习算法 (不考虑用户类型)
class SingleMeanFieldAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Q网络和目标网络
        self.q_net = SingleMeanFieldQNetwork(state_dim, action_dim)
        self.target_q_net = SingleMeanFieldQNetwork(state_dim, action_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.learning_rate)
        self.memory = ReplayBuffer(config.memory_size)

        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay

        # 温度参数
        self.temperature = 0.1

    def get_action(self, state, single_mean, diff_type_mean=None, exploration=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        single_mean_tensor = torch.FloatTensor([single_mean])

        with torch.no_grad():
            q_values = self.q_net(state_tensor, single_mean_tensor)
            probabilities = torch.softmax(q_values / self.temperature, dim=-1)
            action = torch.argmax(probabilities, dim=-1).item()

        return action

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return 0

        # 采样批次数据
        states, actions, rewards, next_states, dones, same_type_means, diff_type_means = self.memory.sample(batch_size)

        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        single_means = torch.FloatTensor(same_type_means)  # 使用same_type_means作为单一平均场

        # 计算当前Q值
        current_q_values = self.q_net(states, single_means)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states, single_means)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + config.gamma * max_next_q_values * (~dones)

        # 计算损失
        loss = torch.mean((target_q_values - current_q_values) ** 2)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 衰减epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def soft_update(self):
        # 软更新目标网络
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - config.target_network_tau) + param.data * config.target_network_tau)

    def push(self, state, action, reward, next_state, done, same_type_mean, diff_type_mean):
        self.memory.push(state, action, reward, next_state, done, same_type_mean, diff_type_mean)


# 完全可观测平均场Q学习算法
class FullObsMeanFieldAgent:
    def __init__(self, global_state_dim, action_dim):
        self.global_state_dim = global_state_dim
        self.action_dim = action_dim

        # Q网络和目标网络
        self.q_net = FullObsMeanFieldQNetwork(global_state_dim, action_dim)
        self.target_q_net = FullObsMeanFieldQNetwork(global_state_dim, action_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.learning_rate)
        self.memory = ReplayBuffer(config.memory_size)

        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay

        # 温度参数
        self.temperature = 0.1

    def get_action(self, global_state, same_type_mean, diff_type_mean, exploration=True):
        global_state_tensor = torch.FloatTensor(global_state).unsqueeze(0)
        same_type_tensor = torch.FloatTensor([same_type_mean])
        diff_type_tensor = torch.FloatTensor([diff_type_mean])

        with torch.no_grad():
            q_values = self.q_net(global_state_tensor, same_type_tensor, diff_type_tensor)
            probabilities = torch.softmax(q_values / self.temperature, dim=-1)
            action = torch.argmax(probabilities, dim=-1).item()

        return action

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return 0

        # 采样批次数据
        states, actions, rewards, next_states, dones, same_type_means, diff_type_means = self.memory.sample(batch_size)

        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        same_type_means = torch.FloatTensor(same_type_means)
        diff_type_means = torch.FloatTensor(diff_type_means)

        # 计算当前Q值
        current_q_values = self.q_net(states, same_type_means, diff_type_means)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states, same_type_means, diff_type_means)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + config.gamma * max_next_q_values * (~dones)

        # 计算损失
        loss = torch.mean((target_q_values - current_q_values) ** 2)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 衰减epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def soft_update(self):
        # 软更新目标网络
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - config.target_network_tau) + param.data * config.target_network_tau)

    def push(self, state, action, reward, next_state, done, same_type_mean, diff_type_mean):
        self.memory.push(state, action, reward, next_state, done, same_type_mean, diff_type_mean)


# POMF Actor-Critic算法
class POMFACAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor网络和Critic网络
        self.actor = POMFACActor(state_dim, action_dim)
        self.critic = POMFACCritic(state_dim, action_dim)

        # 目标网络
        self.target_actor = POMFACActor(state_dim, action_dim)
        self.target_critic = POMFACCritic(state_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.pomf_ac_actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.pomf_ac_critic_lr)

        # 经验回放缓冲区
        self.memory = ReplayBuffer(config.memory_size)

        # 探索参数
        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay

    def get_action(self, state, same_type_mean, diff_type_mean, exploration=True):
        if exploration and random.random() < self.epsilon:
            # epsilon-greedy探索
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            same_type_tensor = torch.FloatTensor([same_type_mean])
            diff_type_tensor = torch.FloatTensor([diff_type_mean])

            with torch.no_grad():
                policy = self.actor(state_tensor, same_type_tensor, diff_type_tensor)
                # 根据策略概率分布采样动作
                action_probs = policy.cpu().numpy()[0]
                action = np.random.choice(self.action_dim, p=action_probs)

            return action

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return 0, 0

        # 采样批次数据
        states, actions, rewards, next_states, dones, same_type_means, diff_type_means = self.memory.sample(batch_size)

        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        same_type_means = torch.FloatTensor(same_type_means)
        diff_type_means = torch.FloatTensor(diff_type_means)

        batch_size_t = states.shape[0]

        # 将动作转换为one-hot编码
        actions_one_hot = torch.zeros(batch_size_t, self.action_dim)
        actions_one_hot.scatter_(1, actions.unsqueeze(1), 1)

        # 更新Critic网络
        with torch.no_grad():
            # 计算目标动作
            next_policy = self.target_actor(next_states, same_type_means, diff_type_means)

            # 计算目标Q值
            next_actions_one_hot = torch.zeros(batch_size_t, self.action_dim)
            next_actions = torch.argmax(next_policy, dim=1)
            next_actions_one_hot.scatter_(1, next_actions.unsqueeze(1), 1)

            target_q_values = self.target_critic(next_states, next_actions_one_hot, same_type_means,
                                                 diff_type_means).squeeze()
            target_q_values = rewards + config.gamma * target_q_values * (~dones)

        # 计算当前Q值
        current_q_values = self.critic(states, actions_one_hot, same_type_means, diff_type_means).squeeze()

        # Critic损失
        critic_loss = F.mse_loss(current_q_values, target_q_values)

        # 优化Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # 更新Actor网络
        # 计算当前策略
        current_policy = self.actor(states, same_type_means, diff_type_means)

        # 使用Critic评估当前策略
        current_actions_one_hot = torch.zeros(batch_size_t, self.action_dim)
        current_actions = torch.argmax(current_policy, dim=1)
        current_actions_one_hot.scatter_(1, current_actions.unsqueeze(1), 1)

        actor_q_values = self.critic(states, current_actions_one_hot, same_type_means, diff_type_means).squeeze()

       
        actor_loss = -actor_q_values.mean()

      
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # 软更新目标网络
        self.soft_update()

        # 衰减epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return actor_loss.item(), critic_loss.item()

    def soft_update(self):
        # 软更新目标网络
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - config.target_network_tau) + param.data * config.target_network_tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - config.target_network_tau) + param.data * config.target_network_tau)

    def push(self, state, action, reward, next_state, done, same_type_mean, diff_type_mean):
        self.memory.push(state, action, reward, next_state, done, same_type_mean, diff_type_mean)


# 训练函数
def train_algorithm(env, agents, num_episodes, algorithm_type):
    episode_rewards = []
    episode_latency=[]
    episode_energy=[]
    episode_losses = []
    start = time.perf_counter()
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        total_rewards = np.zeros(env.num_users)
        # total_latency_rewards=np.zeros(len([i for i, agent in enumerate(agents) if agent['type'] == UserType.LATENCY_SENSITIVE]))
        # total_energy_rewards = np.zeros(len([i for i, agent in enumerate(agents) if agent['type'] == UserType.ENERGY_SENSITIVE]))
        total_loss = 0
        update_count = 0

      
        if algorithm_type in ["MADDPG", "FullObsMeanField"]:
            global_state = env.get_global_state()

        for step in range(config.max_steps):
            actions = []
            same_type_means = []
            diff_type_means = []
            single_means = []

            # 获取每个智能体的动作
            for i, agent in enumerate(agents):
                if algorithm_type == "POMFQ":
                    same_type_mean, diff_type_mean = env.get_mean_field_actions(i)
                    same_type_means.append(same_type_mean)
                    diff_type_means.append(diff_type_mean)
                    action = agent.get_action(state[i], same_type_mean, diff_type_mean, exploration=True)
                elif algorithm_type == "IQL":
                    action = agent.get_action(state[i], exploration=True)
                elif algorithm_type == "MADDPG":
                    action = agent.get_action(state[i], exploration=True)
                elif algorithm_type == "SingleMeanField":
                    single_mean = env.get_single_mean_field_actions(i)
                    single_means.append(single_mean)
                    action = agent.get_action(state[i], single_mean, exploration=True)
                elif algorithm_type == "FullObsMeanField":
                    same_type_mean, diff_type_mean = env.get_mean_field_actions(i)
                    same_type_means.append(same_type_mean)
                    diff_type_means.append(diff_type_mean)
                    action = agent.get_action(global_state, same_type_mean, diff_type_mean, exploration=True)
                elif algorithm_type == "POMFAC": 
                    same_type_mean, diff_type_mean = env.get_mean_field_actions(i)
                    same_type_means.append(same_type_mean)
                    diff_type_means.append(diff_type_mean)
                    action = agent.get_action(state[i], same_type_mean, diff_type_mean, exploration=True)
                else: 
                    action = agent.get_action(state[i], exploration=True)

                actions.append(action)

            # 执行动作
            next_state, rewards, done = env.step(actions)

            # 存储经验
            if algorithm_type == "POMFQ":
                for i, agent in enumerate(agents):
                    agent.memory.push(state[i], actions[i], rewards[i], next_state[i],
                                      done, same_type_means[i], diff_type_means[i])
            elif algorithm_type == "IQL":
                for i, agent in enumerate(agents):
                    agent.push(state[i], actions[i], rewards[i], next_state[i], done)
            elif algorithm_type == "MADDPG":
           
                # next_global_state = env.get_global_state()
                # agents[0].push(state, actions, rewards, next_state, [done] * env.num_users)
                # global_state = next_global_state
                all_states = np.concatenate(state)
                all_next_states = np.concatenate(next_state)

            
                # Critic期望的动作输入是每个智能体的动作概率分布
                all_actions = []
                for i, action in enumerate(actions):
                    # 创建动作的概率分布
                    action_probs = np.zeros(agents[i].action_dim)
                    action_probs[action] = 1.0  # one-hot编码
                    all_actions.extend(action_probs)
                all_actions = np.array(all_actions)

                # 存储经验
                agents[0].push(all_states, all_actions, rewards, all_next_states, [done] * env.num_users)
            elif algorithm_type == "SingleMeanField":
                for i, agent in enumerate(agents):
                    agent.push(state[i], actions[i], rewards[i], next_state[i],
                               done, single_means[i], 0)  # 第二个平均场参数设为0
            elif algorithm_type == "FullObsMeanField":
                next_global_state = env.get_global_state()
                for i, agent in enumerate(agents):
                    agent.push(global_state, actions[i], rewards[i], next_global_state,
                               done, same_type_means[i], diff_type_means[i])
                global_state = next_global_state
            elif algorithm_type == "POMFAC": 
                for i, agent in enumerate(agents):
                    agent.push(state[i], actions[i], rewards[i], next_state[i],
                               done, same_type_means[i], diff_type_means[i])

            # 更新网络
            if algorithm_type == "MADDPG":
              
                for i, agent in enumerate(agents):
                    actor_loss, critic_loss = agent.update(config.batch_size, agents)
                    if actor_loss > 0:
                        total_loss += actor_loss + critic_loss
                        update_count += 1
            elif algorithm_type == "POMFAC": 
                for agent in agents:
                    actor_loss, critic_loss = agent.update(config.batch_size)
                    if actor_loss > 0:
                        total_loss += actor_loss + critic_loss
                        update_count += 1
            else:
                for agent in agents:
                    if hasattr(agent, 'update'):
                        loss = agent.update(config.batch_size)
                        if loss > 0:
                            total_loss += loss
                            update_count += 1
                        if hasattr(agent, 'soft_update'):
                            agent.soft_update()

            state = next_state
            total_rewards += rewards

            if done:
                break

        avg_reward = np.mean(total_rewards)
        total_latency_rewards=[total_rewards[i] for i,user in enumerate(env.users) if user['type']==UserType.LATENCY_SENSITIVE]
        total_energy_rewards = [total_rewards[i] for i, user in enumerate(env.users) if
                                 user['type'] == UserType.ENERGY_SENSITIVE]
        avg_latency_reward=np.mean(total_latency_rewards)
        avg_energy_reward=np.mean(total_energy_rewards)
        avg_loss = total_loss / update_count if update_count > 0 else 0

        episode_rewards.append(avg_reward)
        episode_latency.append(avg_latency_reward)
        episode_energy.append(avg_energy_reward)
        episode_losses.append(avg_loss)

        if episode % 100 == 0:
            print(f"Episode {episode}, Average Reward: {avg_reward:.4f}, Average Latency Reward: {avg_latency_reward:.4f}, Average Energy Reward: {avg_energy_reward:.4f}, Average Loss: {avg_loss:.4f}")
    end = time.perf_counter()
    run_time = end - start
    print(f"the {algorithm_type} training time is {run_time}")
    return episode_rewards,episode_latency,episode_energy, episode_losses


# 测试函数
def test_algorithm(env, agents, algorithm_type):
    state = env.reset()
    total_rewards = np.zeros(env.num_users)
    total_delays = []
    total_energies = []
    latency_delays = []
    energy_energies = []

  
    if algorithm_type in ["MADDPG", "FullObsMeanField"]:
        global_state = env.get_global_state()

    for step in range(config.max_steps):
        actions = []
        same_type_means = []
        diff_type_means = []
        single_means = []

        for i, agent in enumerate(agents):
            if algorithm_type == "POMFQ":
                same_type_mean, diff_type_mean = env.get_mean_field_actions(i)
                same_type_means.append(same_type_mean)
                diff_type_means.append(diff_type_mean)

                action = agent.get_action(state[i], same_type_mean, diff_type_mean, exploration=False)

            elif algorithm_type == "IQL":
                action = agent.get_action(state[i], exploration=False)
            elif algorithm_type == "MADDPG":
                action = agent.get_action(state[i], exploration=False)
            elif algorithm_type == "SingleMeanField":
                single_mean = env.get_single_mean_field_actions(i)
                single_means.append(single_mean)
                action = agent.get_action(state[i], single_mean, exploration=False)
            elif algorithm_type == "FullObsMeanField":
                same_type_mean, diff_type_mean = env.get_mean_field_actions(i)
                same_type_means.append(same_type_mean)
                diff_type_means.append(diff_type_mean)
                action = agent.get_action(global_state, same_type_mean, diff_type_mean, exploration=False)
            elif algorithm_type == "POMFAC": 
                same_type_mean, diff_type_mean = env.get_mean_field_actions(i)
                same_type_means.append(same_type_mean)
                diff_type_means.append(diff_type_mean)
                action = agent.get_action(state[i], same_type_mean, diff_type_mean, exploration=False)
            else: 
                action = agent.get_action(state[i], exploration=False)
            actions.append(action)

        next_state, rewards, done = env.step(actions)
      
        if algorithm_type in ["MADDPG", "FullObsMeanField"]:
            next_global_state = env.get_global_state()
            global_state = next_global_state

        state = next_state
        total_rewards += rewards
        for i, user in enumerate(env.users):
            task = user['current_task']
            if actions[i] == 0:  
                exec_time = task.computation_requirement / user['f_i_l']
                delay = exec_time + user['local_waiting_time']
                energy = config.k * (user['f_i_l'] ** 2) * task.computation_requirement
            else:  
                transmission_rate = env.calculate_transmission_rate(user)
                transmission_time = task.data_size / transmission_rate
                active_users = env.get_active_users_count()
                mec_exec_time = (task.computation_requirement / config.F_e) * active_users
                delay = (transmission_time + user['transmission_waiting_time'] +
                         mec_exec_time + user['mec_waiting_time'])
                energy = user['p_i'] * transmission_time

            total_delays.append(delay)
            total_energies.append(energy)

   

        for i, user in enumerate(env.users):
            task = user['current_task']
            if user['type'] == UserType.LATENCY_SENSITIVE:
                if actions[i] == 0:
                    exec_time = task.computation_requirement / user['f_i_l']
                    delay = exec_time + user['local_waiting_time']
                else:
                    transmission_rate = env.calculate_transmission_rate(user)
                    transmission_time = task.data_size / transmission_rate
                    active_users = env.get_active_users_count()
                    mec_exec_time = (task.computation_requirement / config.F_e) * active_users
                    delay = (transmission_time + user['transmission_waiting_time'] +
                             mec_exec_time + user['mec_waiting_time'])
                latency_delays.append(delay)
            else:
                if actions[i] == 0:
                    energy = config.k * (user['f_i_l'] ** 2) * task.computation_requirement
                else:
                    transmission_rate = env.calculate_transmission_rate(user)
                    transmission_time = task.data_size / transmission_rate
                    energy = user['p_i'] * transmission_time
                energy_energies.append(energy)

        if done:
            break

    avg_reward = np.mean(total_rewards)
    avg_delay = np.mean(total_delays) if total_delays else 0
    avg_energy = np.mean(total_energies) if total_energies else 0

    avg_latency_delay = np.mean(latency_delays) if latency_delays else 0
    avg_energy_energy = np.mean(energy_energies) if energy_energies else 0

    return (avg_reward, avg_delay, avg_energy,
            avg_latency_delay, avg_energy_energy)


def visualize_user_locations(env, episode):
    plt.figure(figsize=(10, 8))

    # 绘制用户位置
    for user in env.users:
        x, y = user['location']
        color = 'red' if user['type'] == UserType.LATENCY_SENSITIVE else 'blue'
        marker = 'o' if user['type'] == UserType.LATENCY_SENSITIVE else 's'
        plt.scatter(x, y, c=color, marker=marker, s=100,
                    label=f"{'Latency' if user['type'] == UserType.LATENCY_SENSITIVE else 'Energy'}-sensitive")

       
        for neighbor_id in user['neighbors']:
            neighbor = env.users[neighbor_id]
            nx, ny = neighbor['location']
            plt.plot([x, nx], [y, ny], 'gray', alpha=0.3, linewidth=0.5)

   
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title(f'User Locations and Neighbor Relationships (Episode {episode})')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, config.area_size)
    plt.ylim(0, config.area_size)
    import os
    plt.savefig(os.path.join(os.getcwd(), 'user_locations_episode_%s.png' % episode), dpi=300, bbox_inches='tight')
    # plt.show()
    # plt.close()


# 主函数
def main():
    # 参数设置
    num_users = config.N
    num_episodes = config.num_episodes

    # 创建环境
    env = MECEnvironment(num_users)
    state_dim = len(env.users[0]['local_observation'])
    action_dim = 2
    global_state_dim = state_dim * num_users

    # 可视化初始用户位置
    visualize_user_locations(env, 0)

    # 定义算法列表
    algorithms = [
        "POMFQ", "IQL", "MADDPG", "SingleMeanField", "FullObsMeanField", "POMFAC",
        "Random", "Local", "Offload"
    ]

    all_results = {}

    # 训练和测试每种算法
    for algo in algorithms:
        print(f"\nTraining {algo} algorithm...")

        # 创建智能体
        if algo == "POMFQ":
            agents = [POMFQLearningAgent(state_dim, action_dim) for _ in range(num_users)]
            rewards,latency_rewards,energy_rewards, losses = train_algorithm(env, agents, num_episodes, algo)
        elif algo == "IQL":
            agents = [IQLAgent(state_dim, action_dim) for _ in range(num_users)]
            rewards,latency_rewards,energy_rewards, losses = train_algorithm(env, agents, num_episodes, algo)
        elif algo == "MADDPG":
            agents = [MADDPGAgent(state_dim, action_dim, num_users, i) for i in range(num_users)]
            rewards,latency_rewards,energy_rewards, losses = train_algorithm(env, agents, num_episodes, algo)
        elif algo == "SingleMeanField":
            agents = [SingleMeanFieldAgent(state_dim, action_dim) for _ in range(num_users)]
            rewards,latency_rewards,energy_rewards, losses = train_algorithm(env, agents, num_episodes, algo)
        elif algo == "FullObsMeanField":
            agents = [FullObsMeanFieldAgent(global_state_dim, action_dim) for _ in range(num_users)]
            rewards,latency_rewards,energy_rewards, losses = train_algorithm(env, agents, num_episodes, algo)
        elif algo == "POMFAC": 
            agents = [POMFACAgent(state_dim, action_dim) for _ in range(num_users)]
            rewards,latency_rewards,energy_rewards, losses = train_algorithm(env, agents, num_episodes, algo)
        elif algo == "Random":
            agents = [RandomAgent() for _ in range(num_users)]
            rewards, losses = [0], [0] 
        elif algo == "Local":
            agents = [LocalExecutionAgent() for _ in range(num_users)]
            rewards, losses = [0], [0]
        elif algo == "Offload":
            agents = [OffloadingAgent() for _ in range(num_users)]
            rewards, losses = [0], [0]

        # 测试算法
        print(f"Testing {algo} algorithm...")
        (reward, delay, energy,
         latency_delay, energy_energy) = test_algorithm(env, agents, algo)

       
        all_results[algo] = {
            'reward': reward,
            'delay': delay,
            'energy': energy,
            'latency_delay': latency_delay,
            'energy_energy': energy_energy,
            'training_rewards': rewards,
            'training_latency_rewards':latency_rewards,
            'training_energy_rewards':energy_rewards,
            'training_losses': losses
        }

    # 可视化训练后的用户位置
    visualize_user_locations(env, num_episodes)

   
    print("Testing comparison algorithms...")

    # 可视化结果
    plt.figure(figsize=(20, 15))

    # 训练曲线
    trainable_algorithms = ["POMFQ", "IQL", "MADDPG", "SingleMeanField", "FullObsMeanField", "POMFAC"]

    # 训练奖励曲线
    plt.subplot(2, 2, 1)
    for algo in trainable_algorithms:
        if algo in all_results:
            plt.plot(all_results[algo]['training_rewards'], label=algo)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()

    plt.subplot(2, 2, 2)
    for algo in trainable_algorithms:
        if algo in all_results:
            plt.plot(all_results[algo]['training_latency_rewards'], label=algo)
    plt.title('Training Latency Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average Latency Reward')
    plt.legend()

    plt.subplot(2, 2, 3)
    for algo in trainable_algorithms:
        if algo in all_results:
            plt.plot(all_results[algo]['training_energy_rewards'], label=algo)
    plt.title('Training Energy Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average Energy Reward')
    plt.legend()

    # 训练损失曲线
    plt.subplot(2, 2, 4)
    for algo in trainable_algorithms:
        if algo in all_results and len(all_results[algo]['training_losses']) > 0:
            plt.plot(all_results[algo]['training_losses'], label=algo)
    plt.title('Training Losses')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.legend()

    plt.tight_layout()
    import os
    plt.savefig(os.path.join(os.getcwd(), 'training_results.png'), dpi=300)
    #plt.show()

    # 算法对比 - 奖励
    plt.subplot(2, 3, 1)
    rewards = [all_results[algo]['reward'] for algo in algorithms]
    plt.bar(algorithms, rewards)
    plt.title('Algorithm Comparison - Rewards')
    plt.ylabel('Average Reward')
    plt.xticks(rotation=45)

    # 算法对比 - 延迟
    plt.subplot(2, 3, 2)
    delays = [all_results[algo]['delay'] for algo in algorithms]
    plt.bar(algorithms, delays)
    plt.title('Algorithm Comparison - Delays')
    plt.ylabel('Average Delay')
    plt.xticks(rotation=45)

    # 算法对比 - 能耗
    plt.subplot(2, 3, 3)
    energies = [all_results[algo]['energy'] for algo in algorithms]
    plt.bar(algorithms, energies)
    plt.title('Algorithm Comparison - Energies')
    plt.ylabel('Average Energy')
    plt.xticks(rotation=45)

    # 用户类型性能 - 延迟敏感型
    plt.subplot(2, 3, 4)
    latency_delays = [all_results[algo]['latency_delay'] for algo in algorithms]
    plt.bar(algorithms, latency_delays)
    plt.title('Latency-sensitive Users - Delays')
    plt.ylabel('Average Task Delay')
    plt.xticks(rotation=45)

    # 用户类型性能 - 能耗敏感型
    plt.subplot(2, 3, 5)
    energy_energies = [all_results[algo]['energy_energy'] for algo in algorithms]
    plt.bar(algorithms, energy_energies)
    plt.title('Energy-sensitive Users - Energies')
    plt.ylabel('Average Task Energy')
    plt.xticks(rotation=45)

    plt.tight_layout()
    import os
    plt.savefig(os.path.join(os.getcwd(), 'comparison_results.png'), dpi=300, bbox_inches='tight')
    #plt.show()

    # 保存结果
    results_df = pd.DataFrame({
        'Algorithm': algorithms,
        'Reward': [all_results[algo]['reward'] for algo in algorithms],
        'Delay': [all_results[algo]['delay'] for algo in algorithms],
        'Energy': [all_results[algo]['energy'] for algo in algorithms],
        'LatencyDelay': [all_results[algo]['latency_delay'] for algo in algorithms],
        'EnergyConsumption': [all_results[algo]['energy_energy'] for algo in algorithms]
    })
    results_df.to_csv(os.path.join(os.getcwd(), 'comparison_results.csv'), index=False)
    print("Results saved to comparison_results.csv")


if __name__ == "__main__":
    main()