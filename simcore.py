import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
from functools import cmp_to_key
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
import threading
import time
import multiprocessing
from multiprocessing import Manager
import cProfile
import heapq
import math

# ==== 新增：RL & 可视化相关 ====
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from dataclasses import dataclass
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os

# ==== 调度追踪相关数据类 ====
@dataclass
class ScheduleRecord:
    """单步调度记录"""
    timestamp: float           # 调度时刻
    sched_uid: int            # 调度的用户ID (-1表示无调度)
    sched_tid: int            # 调度的业务类型 (-1表示无调度)
    queue_sizes: np.ndarray   # [user_num, 3] 所有队列的tot_size
    queue_delays: np.ndarray  # [user_num, 3] 所有队列的头部时延
    max_vo_delay: float       # VO业务最大时延
    max_vi_delay: float       # VI业务最大时延
    max_be_delay: float       # BE业务最大时延

class ScheduleTracker:
    """追踪调度过程的所有细节"""
    def __init__(self, name: str = "tracker"):
        self.name = name
        self.records: List[ScheduleRecord] = []
    
    def add_record(self, record: ScheduleRecord):
        """添加一条调度记录"""
        self.records.append(record)
    
    def get_timestamps(self) -> np.ndarray:
        """获取所有时间戳"""
        return np.array([r.timestamp for r in self.records])
    
    def get_max_delays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取各业务类型的最大时延序列"""
        vo_delays = np.array([r.max_vo_delay for r in self.records])
        vi_delays = np.array([r.max_vi_delay for r in self.records])
        be_delays = np.array([r.max_be_delay for r in self.records])
        return vo_delays, vi_delays, be_delays
    
    def get_total_queue_sizes(self) -> np.ndarray:
        """获取总队列流量序列"""
        return np.array([r.queue_sizes.sum() for r in self.records])
    
    def get_schedule_decisions(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取调度决策序列 (uid, tid)"""
        uids = np.array([r.sched_uid for r in self.records])
        tids = np.array([r.sched_tid for r in self.records])
        return uids, tids
    
    def get_queue_size_by_user_tid(self, uid: int, tid: int) -> np.ndarray:
        """获取特定用户和业务类型的队列流量序列"""
        return np.array([r.queue_sizes[uid, tid] for r in self.records])
    
    def summary(self) -> Dict[str, Any]:
        """生成统计摘要"""
        if not self.records:
            return {"name": self.name, "total_steps": 0}
        
        vo_delays, vi_delays, be_delays = self.get_max_delays()
        total_sizes = self.get_total_queue_sizes()
        uids, tids = self.get_schedule_decisions()
        
        # 统计调度分布
        schedule_dist = {}
        for uid, tid in zip(uids, tids):
            if uid == -1:
                key = "no_schedule"
            else:
                key = f"AC{tid}"
            schedule_dist[key] = schedule_dist.get(key, 0) + 1
        
        return {
            "name": self.name,
            "total_steps": len(self.records),
            "avg_vo_delay": float(vo_delays.mean()),
            "max_vo_delay": float(vo_delays.max()),
            "avg_vi_delay": float(vi_delays.mean()),
            "max_vi_delay": float(vi_delays.max()),
            "avg_be_delay": float(be_delays.mean()),
            "max_be_delay": float(be_delays.max()),
            "avg_total_queue": float(total_sizes.mean()),
            "max_total_queue": float(total_sizes.max()),
            "schedule_distribution": schedule_dist,
        }

class Task:
    def __init__(self, event, timestamp):
        self.event = event
        self.timestamp = timestamp
    def __lt__(self, other):
        # 定义小于号的比较方式，用于 heapq 排序
        return self.timestamp < other.timestamp
    def __repr__(self):
        return f"Task(name={self.name}, priority={self.priority})"
def compare(a, b):
    if a["rate"] < b["rate"]:
        return 1
    return -1
def cmp(a, b):
    if a["timestamp"] > b["timestamp"]:
        return 1
    elif a["timestamp"] == b["timestamp"]:
        if a["size"] < b["size"]:
            return 1
        else:
            return -1
    else:
       return -1
def create_txq_table(user_num, queue_size):
    txq_table = []
    # 创建一个线程安全的队列
    for user_id in range(user_num):
        tid_table = []
        for tid in range(3):
            #这里使用环形队列实现
            tid_table.append([0 for _ in range(queue_size)])
        txq_table.append(tid_table)
    return txq_table
def create_tx_info_table(user_num):
    tx_info_table = []
    # 创建一个线程安全的队列
    for user_id in range(user_num):
        tx_info_tid_table = []
        for tid in range(3):
            tx_info_tid_table.append({"tot_size" : 0, "read_ptr" : 0, "write_ptr" : 0})
        tx_info_table.append(tx_info_tid_table)
    return tx_info_table
   
def create_rate_table(user_num):
    rate_table = []
    max_band_width = 2442
    ofo_list = []
    for user_id in range(user_num):
        tid_rate_table = []
        for tid in range(3):
            if tid == 0:
                rand_rate = random.randint(1, 20)
            elif tid == 1:
                rand_rate = random.randint(1, 40)
            elif tid == 2:
                rand_rate = random.randint(1, 100)
            #ratetable包含两个信息，速率和发包间隔
            #tid_rate_table.append(rand_rate * 1000 * 1000)
            if tid == 0:
                rand_interval = random.randint(1, 20)
                #tid_rate_table.append(rand_interval)
            elif tid == 1:
                rand_interval = random.randint(1, 10)
                #tid_rate_table.append(rand_interval)
            else:
                rand_interval = random.randint(1, 3)
                #tid_rate_table.append(rand_interval)
            rate_info = {
                'uid' : user_id,
                'tid' : tid,
                'rate' : rand_rate
            }
            tid_rate_table.append([0, rand_interval])
            ofo_list.append(rate_info)
        #速率先写入0
        rate_table.append(tid_rate_table)
    sorted_list = sorted(ofo_list, key=cmp_to_key(compare))
   
    sum_rate = 0
    for i in range(len(sorted_list)):
        if sum_rate + sorted_list[i]['rate'] < max_band_width:
            uid = sorted_list[i]['uid']
            tid = sorted_list[i]['tid']
            rate_table[uid][tid][0] = sorted_list[i]['rate'] * 1000 * 1000
            #print(f"user {uid}, ac {tid}, set rate is {sorted_list[i]['rate']}")
            sum_rate = sum_rate + sorted_list[i]['rate']
    #调试用
    '''
    for i in range(user_num):
        for j in range(3):
            print(f"user {i}, ac {j}, rate is {rate_table[i][j][0]}")
    '''
    return rate_table
def gen_sched_res(txq_table, tx_info_table, user_num, timestamp_now, queue_size):
    vo_list = []
    vi_list = []
    be_list = []
    #首先从txq_table中取首包做排序
    schedule_res = {"uid" : -1, "ac_type" : -1}
    for uid in range(user_num):
        for ac in range(3):
            w_ptr = tx_info_table[uid][ac]["write_ptr"]
            r_ptr = tx_info_table[uid][ac]["read_ptr"]
            if w_ptr != r_ptr: #队列非空
                pkt_info = txq_table[uid][ac][r_ptr]
                #txq_table[uid][ac].pop(0)
                schedule_info = {
                    "uid" : uid,
                    "ac_type" : ac,
                    "size" : tx_info_table[uid][ac]["tot_size"],
                    "timestamp" : pkt_info['timestamp']
                }
                if ac == 0:
                    vo_list.append(schedule_info)
                elif ac == 1:
                    vi_list.append(schedule_info)
                else:
                    be_list.append(schedule_info)
               
    if len(vo_list) == 0 and len(vi_list) == 0 and len(be_list) == 0:
        schedule_res = {"uid" : -1, "ac_type" : -1}
        return schedule_res
    sorted_vo_list = sorted(vo_list, key=cmp_to_key(cmp))
    sorted_vi_list = sorted(vi_list, key=cmp_to_key(cmp))
    sorted_be_list = sorted(be_list, key=cmp_to_key(cmp))
    #尝试调度VO
    #print(f"timestamp {timestamp_now} vo queue length is {len(sorted_vo_list)}")
    for i in range(len(sorted_vo_list)):
        if timestamp_now - sorted_vo_list[i]["timestamp"] <= 20 and sorted_vo_list[i]["timestamp"] < timestamp_now:
            #print(f"schedule :: uid is {(sorted_vo_list[i]['uid'])}, ac type is vo")
            schedule_res = {"uid" : sorted_vo_list[i]['uid'], "ac_type" : 0}
            return schedule_res
        elif timestamp_now - sorted_vo_list[i]["timestamp"] > 20:
            #丢弃超时报文
            uid = sorted_vo_list[i]['uid']
            r_ptr = tx_info_table[uid][0]["read_ptr"]
            w_ptr = tx_info_table[uid][0]["write_ptr"]
            while r_ptr != w_ptr:
                if timestamp_now - txq_table[uid][0][r_ptr]['timestamp'] > 20:
                    #记录丢弃长度
                    tx_info_table[uid][0]["tot_size"] = tx_info_table[uid][0]["tot_size"] - txq_table[uid][0][r_ptr]['size']
                    r_ptr = (r_ptr + 1) % queue_size
                else:
                    break
            tx_info_table[uid][0]["read_ptr"] = r_ptr
    #尝试调度VI
    for i in range(len(sorted_vi_list)):
        if timestamp_now - sorted_vi_list[i]["timestamp"] > 20 and timestamp_now - sorted_vi_list[i]["timestamp"] <= 50 and sorted_vi_list[i]["timestamp"] < timestamp_now:
            #print(f"schedule :: uid is {(sorted_vo_list[i]['uid'])}, ac type is vi")
            schedule_res = {"uid" : sorted_vi_list[i]['uid'], "ac_type" : 1}
            return schedule_res
        elif timestamp_now - sorted_vi_list[i]["timestamp"] > 50:
            #丢弃超时报文
            uid = sorted_vo_list[i]['uid']
            r_ptr = tx_info_table[uid][1]["read_ptr"]
            w_ptr = tx_info_table[uid][1]["write_ptr"]
            while r_ptr != w_ptr:
                if timestamp_now - txq_table[uid][1][r_ptr]['timestamp'] > 50:
                    #记录丢弃长度
                    tx_info_table[uid][1]["tot_size"] = tx_info_table[uid][1]["tot_size"] - txq_table[uid][1][r_ptr]['size']
                    r_ptr = (r_ptr + 1) % queue_size
                else:
                    break
            tx_info_table[uid][1]["read_ptr"] = r_ptr
    #调度BE，假定所有用户的最小需求都是100ms调度一次
    for i in range(len(sorted_be_list)):
        if timestamp_now - sorted_be_list[i]["timestamp"] > 100 and sorted_be_list[i]["timestamp"] < timestamp_now:
            #print(f"schedule :: uid is {(sorted_be_list[i]['uid'])}, ac type is be")
            schedule_res = {"uid" : sorted_be_list[i]['uid'], "ac_type" : 2}
            return schedule_res
    #没有调度到BE，调度剩余报文最长的用户
    max_pkt_size = 0
    res_uid = -1
    res_ac_type = -1
    #这个时候队列可能已经发生变化
    for i in range(user_num):
        if tx_info_table[i][1]["tot_size"] > max_pkt_size:
            res_uid = i
            max_pkt_size = tx_info_table[i][1]["tot_size"]
            res_ac_type = 1
    for i in range(user_num):
        if tx_info_table[i][2]["tot_size"] > max_pkt_size:
            res_uid = i
            max_pkt_size = tx_info_table[i][2]["tot_size"]
            res_ac_type = 2
    #print(f"schedule :: uid is {res_uid}, ac type is {res_ac_type}, pkt_size = {max_pkt_size}")
    #print(f"get_schres:: r_ptr is {tx_info_table[res_uid][res_ac_type]['read_ptr']}, w_ptr is {tx_info_table[res_uid][res_ac_type]['write_ptr']}, remain_size is {tx_info_table[res_uid][res_ac_type]['tot_size']}")
    schedule_res = {"uid" : res_uid, "ac_type" : res_ac_type}
    return schedule_res
   
def schedule(txq_table, tx_info_table, user_num, timestamp, queue_size, event_queue):
    max_rate = 2442 * 1000 * 1000
    #先尝试调度VO
    #all_empty = True
    #print("call gen_sched_res")
    sched_res = gen_sched_res(txq_table, tx_info_table, user_num, timestamp, queue_size)
    '''
    globals_dict = {
        'txq_table': txq_table,
        'tx_info_table': tx_info_table,
        'user_num': user_num,
        'start_time': start_time,
        'queue_size': queue_size,
        'gen_sched_res': gen_sched_res # 确保函数也在子进程中可用
    }
    # 使用 runctx 代替 run，并传入 globals 和locals
    cProfile.runctx(
        'gen_sched_res(txq_table, tx_info_table, user_num, start_time, queue_size)',
        globals_dict,
        {},
        sort='time'
    )
    '''
    #cProfile.run('gen_sched_res(txq_table, tx_info_table, user_num, start_time, queue_size)')
    sched_uid = sched_res["uid"]
    sched_tid = sched_res["ac_type"]
    if sched_uid == -1:
        print(f"nothing to schedule!!")
        heapq.heappush(event_queue, Task("schedule", timestamp + 1))
        return
    print(f"schedule uid is {sched_uid}, tid is {sched_tid}")
    temp_sched_size = 0
    r_ptr = tx_info_table[sched_uid][sched_tid]["read_ptr"]
    w_ptr = tx_info_table[sched_uid][sched_tid]["write_ptr"]
    #print(f"r_ptr is {r_ptr}, w_ptr is {w_ptr}")
    while r_ptr != w_ptr:
        if temp_sched_size == max_rate / 1000 * 4:
            break
        if temp_sched_size + txq_table[sched_uid][sched_tid][r_ptr]["size"] <= max_rate / 1000 * 4:
            temp_sched_size = temp_sched_size + txq_table[sched_uid][sched_tid][r_ptr]["size"]
            #print(f"r_ptr is {r_ptr}, packet_size is {txq_table[sched_uid][sched_tid][0]['size']}, sched_size is {temp_sched_size}")
            r_ptr = (r_ptr + 1) % queue_size
        else:
            txq_table[sched_uid][sched_tid][r_ptr]["size"] = txq_table[sched_uid][sched_tid][r_ptr]["size"] - max_rate / 1000 * 4 + temp_sched_size
            temp_sched_size = max_rate / 1000 * 4
    tx_info_table[sched_uid][sched_tid]["read_ptr"] = r_ptr
   
    tx_info_table[sched_uid][sched_tid]["tot_size"] = tx_info_table[sched_uid][sched_tid]["tot_size"] - temp_sched_size
    send_time = temp_sched_size / max_rate * 1000
    if send_time == 0:
        import pdb;pdb.set_trace()
    #print(f"schedule complete timestamp is {timestamp}, send_time is {send_time}, temp_sched_size is {temp_sched_size}, queue remain size is {tx_info_table[sched_uid][sched_tid]['tot_size']}")
   
    heapq.heappush(event_queue, Task("schedule", timestamp + send_time))
    #print(f"sleep time is {send_time / 1000}")
def traffic_generator(txq_table, user_num, timestamp, rate_table, tx_info_table, queue_size, event_queue):
    for uid in range(user_num):
        for ac_num in range(3):
            if rate_table[uid][ac_num][0] > 0:
                #速率表中指定了发包间隔
                if timestamp % rate_table[uid][ac_num][1] == 0:
                    if (tx_info_table[uid][ac_num]["write_ptr"] + 1) % queue_size == tx_info_table[uid][ac_num]["read_ptr"]:
                        print(f"time {timestamp} uid {uid}, tid {ac_num} queue is full, drop input packet")
                        continue
                    packet_size = np.random.poisson(lam=rate_table[uid][ac_num][0] * 0.001)
                    pkt_info = {}
                    pkt_info['size'] = packet_size
                    pkt_info['timestamp'] = timestamp
                    #存储当前队列缓存包长度
                    #print(f"traffic generate uid : {uid}, ac_type : {ac_num}, pkt_size : {packet_size}, queue_remain len : {tx_info_table[uid][ac_num]['tot_size']}")
                    tx_info_table[uid][ac_num]["tot_size"] = tx_info_table[uid][ac_num]["tot_size"] + packet_size
                    w_ptr = tx_info_table[uid][ac_num]["write_ptr"]
                    txq_table[uid][ac_num][w_ptr] = pkt_info
                    tx_info_table[uid][ac_num]["write_ptr"] = (tx_info_table[uid][ac_num]["write_ptr"] + 1) % queue_size
    heapq.heappush(event_queue, Task("traffic_generate", timestamp + 1))
#def create_stat_table(user_num):
       
# ==============================
#      PPO 算法与调度环境
# ==============================

@dataclass
class PPOConfig:
    user_num: int = 32
    queue_size: int = 2048            # 加大队列长度，减小溢出的概率
    max_sim_time: float = 1e2         # 每个 episode 的最大仿真时间（毫秒）
    gamma: float = 0.99
    lam: float = 0.95                 # GAE lambda
    clip_eps: float = 0.2
    lr: float = 3e-4
    epochs: int = 100                  # 每个 episode 的 PPO 更新轮数
    rollout_steps: int = 1024         # 每轮收集的步数
    minibatch_size: int = 256
    total_episodes: int = 200
    w_vo: float = 5.0
    w_vi: float = 2.0
    w_be: float = 1.0
    seed: int = 42
    num_envs: int = 1                 # 向量环境个数，>1 时可以扩展为多进程
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reward归一化配置
    # 使用动态baseline：当前最大时延反映流量难度
    # - 训练时：归一化reward，考虑流量难度
    # - 评估时：原始reward，公平比较
    normalize_reward: bool = True


def set_global_seeds(seed: int):
    """统一设置 random / numpy / torch 的随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RLSchedulingEnv:
    """
    强化学习版本的调度环境：
    - 复用原有 create_txq_table / create_tx_info_table / create_rate_table / traffic_generator
    - 调度时可以选择：
      * use_traditional=True：使用原 gen_sched_res 规则
      * use_traditional=False：使用 RL 动作 (uid, tid)
    """

    def __init__(self, config: PPOConfig, use_traditional: bool = False, enable_tracking: bool = False,
                 normalize_reward: bool = True):
        self.cfg = config
        self.user_num = config.user_num
        self.queue_size = config.queue_size
        self.max_sim_time = config.max_sim_time
        self.use_traditional = use_traditional
        self.enable_tracking = enable_tracking
        
        # Reward归一化相关
        self.normalize_reward = normalize_reward
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0
        self.reward_m2 = 0.0  # 用于Welford算法计算running variance

        self.txq_table = None
        self.tx_info_table = None
        self.rate_table = None
        self.event_queue = None
        self.current_time = 0.0
        
        # 追踪器
        self.tracker = ScheduleTracker(name="Traditional" if use_traditional else "PPO") if enable_tracking else None

        # obs 维度：每个队列两个特征 + 当前时间
        self.obs_dim = self.user_num * 3 * 2 + 1
        self.act_dim = self.user_num * 3

    def _normalize_reward(self, raw_reward: float, max_delay: float) -> float:
        """
        归一化reward，结合两种策略：
        1. 相对当前最大时延的归一化（动态baseline）
        2. Running mean/std归一化（使用Welford算法）
        
        使用最大时延作为baseline的好处：
        - 不需要预计算传统调度的reward
        - 反映当前流量的固有难度
        - 计算简单高效
        """
        if not self.normalize_reward:
            return raw_reward
        
        # 更新running statistics（Welford's online algorithm）
        self.reward_count += 1
        delta = raw_reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = raw_reward - self.reward_mean
        self.reward_m2 += delta * delta2
        
        if self.reward_count > 1:
            self.reward_var = self.reward_m2 / (self.reward_count - 1)
        
        # 策略1：相对当前最大时延的归一化
        # max_delay越大，说明当前流量难度越大
        # 我们将reward相对于这个难度进行调整
        if max_delay > 1.0:  # 避免除以接近0的值
            # 难度越大，reward越负，但我们需要考虑难度因素
            difficulty_factor = max_delay / 20.0  # VO阈值20ms作为参考
            adjusted_reward = raw_reward / (1.0 + difficulty_factor)
        else:
            adjusted_reward = raw_reward
        
        # 策略2：Running mean/std归一化
        if self.reward_count > 10:  # 积累一定样本后才归一化
            reward_std = np.sqrt(self.reward_var) if self.reward_var > 0 else 1.0
            normalized_reward = (adjusted_reward - self.reward_mean) / (reward_std + 1e-8)
            # Clip到合理范围，防止极端值
            normalized_reward = np.clip(normalized_reward, -10.0, 10.0)
            return normalized_reward
        else:
            # 初期直接返回调整后的reward
            return adjusted_reward
    
    def reset(self, seed: int = None):
        """环境重置，初始化队列和事件"""
        if seed is not None:
            set_global_seeds(seed)
        # 初始化队列和统计信息
        self.txq_table = create_txq_table(self.user_num, self.queue_size)
        self.tx_info_table = create_tx_info_table(self.user_num)
        self.rate_table = create_rate_table(self.user_num)
        self.event_queue = []
        heapq.heappush(self.event_queue, Task("traffic_generate", 0.0))
        heapq.heappush(self.event_queue, Task("schedule", 0.5))
        self.current_time = 0.0
        self.done = False
        return self._build_observation()

    def _build_observation(self):
        """构造当前状态向量：每个 (uid, tid) 的 tot_size 和头部时延 + 当前时间"""
        features = []
        for uid in range(self.user_num):
            for tid in range(3):
                info = self.tx_info_table[uid][tid]
                tot_size = info["tot_size"]
                r_ptr = info["read_ptr"]
                w_ptr = info["write_ptr"]
                if r_ptr != w_ptr:
                    head_ts = self.txq_table[uid][tid][r_ptr]['timestamp']
                    delay = max(0.0, self.current_time - head_ts)
                else:
                    delay = 0.0
                features.append(float(tot_size))
                features.append(float(delay))
        # 加上当前时间（也可以归一化）
        features.append(float(self.current_time))
        return np.array(features, dtype=np.float32)

    def _get_queue_snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取当前队列状态快照：sizes[user_num, 3], delays[user_num, 3]"""
        sizes = np.zeros((self.user_num, 3), dtype=np.float32)
        delays = np.zeros((self.user_num, 3), dtype=np.float32)
        
        for uid in range(self.user_num):
            for tid in range(3):
                info = self.tx_info_table[uid][tid]
                sizes[uid, tid] = info["tot_size"]
                
                r_ptr = info["read_ptr"]
                w_ptr = info["write_ptr"]
                if r_ptr != w_ptr:
                    head_ts = self.txq_table[uid][tid][r_ptr]['timestamp']
                    delays[uid, tid] = max(0.0, self.current_time - head_ts)
        
        return sizes, delays
    
    def _compute_max_delays(self):
        """计算三类业务的最大头部时延"""
        max_vo = 0.0
        max_vi = 0.0
        max_be = 0.0
        for uid in range(self.user_num):
            for tid in range(3):
                info = self.tx_info_table[uid][tid]
                r_ptr = info["read_ptr"]
                w_ptr = info["write_ptr"]
                if r_ptr != w_ptr:
                    head_ts = self.txq_table[uid][tid][r_ptr]['timestamp']
                    d = max(0.0, self.current_time - head_ts)
                    if tid == 0:
                        max_vo = max(max_vo, d)
                    elif tid == 1:
                        max_vi = max(max_vi, d)
                    else:
                        max_be = max(max_be, d)
        return max_vo, max_vi, max_be

    def step(self, action: int = 0):
        """
        执行一步：
        1. 从事件队列中推进到下一个 schedule 事件，期间处理所有 traffic_generate
        2. 在 schedule 时刻按传统 or RL 规则进行一次调度
        3. 返回新的状态、reward、done、info
        """
        if self.done:
            raise RuntimeError("Env is done, call reset() first")

        # 1. 先把时间推进到下一个调度时刻
        while True:
            if not self.event_queue:
                self.done = True
                return self._build_observation(), 0.0, True, {}
            task = heapq.heappop(self.event_queue)
            self.current_time = task.timestamp

            # 动态调整速率（保留原逻辑：每 duration_per_time 新生一套 rate_table）
            duration_per_time = 10000
            if math.floor(self.current_time) % duration_per_time == 0:
                self.rate_table = create_rate_table(self.user_num)

            if task.event == "traffic_generate":
                traffic_generator(
                    self.txq_table,
                    self.user_num,
                    task.timestamp,
                    self.rate_table,
                    self.tx_info_table,
                    self.queue_size,
                    self.event_queue,
                )
            elif task.event == "schedule":
                # 到调度时刻，跳出
                break

        # 2. 执行调度，决定 sched_uid / sched_tid
        if self.use_traditional:
            # 传统调度规则：直接用 gen_sched_res
            sched_res = gen_sched_res(
                self.txq_table,
                self.tx_info_table,
                self.user_num,
                self.current_time,
                self.queue_size
            )
            sched_uid = sched_res["uid"]
            sched_tid = sched_res["ac_type"]
        else:
            # RL 动作映射到 (uid, tid)
            idx = int(action)
            if idx < 0 or idx >= self.act_dim:
                idx = 0
            sched_uid = idx // 3
            sched_tid = idx % 3

            # 检查选择的队列是否为空
            info = self.tx_info_table[sched_uid][sched_tid]
            selected_empty_queue = (info["read_ptr"] == info["write_ptr"])
            
            # 如果选择了空队列，施加惩罚并强制选择一个非空队列
            if selected_empty_queue:
                # 方案1：尝试在同一用户的其他队列中找非空的
                found_nonempty = False
                for alt_tid in range(3):
                    alt_info = self.tx_info_table[sched_uid][alt_tid]
                    if alt_info["read_ptr"] != alt_info["write_ptr"]:
                        sched_tid = alt_tid
                        found_nonempty = True
                        break
                
                # 方案2：如果该用户所有队列都空，则遍历其他用户找非空队列
                if not found_nonempty:
                    for uid in range(self.user_num):
                        for tid in range(3):
                            if self.tx_info_table[uid][tid]["read_ptr"] != self.tx_info_table[uid][tid]["write_ptr"]:
                                sched_uid = uid
                                sched_tid = tid
                                found_nonempty = True
                                break
                        if found_nonempty:
                            break
                
                # 如果所有队列都空（极少情况），则标记为无调度
                if not found_nonempty:
                    sched_uid = -1
                    sched_tid = -1

        # 3. 发包逻辑：完全复用原 schedule() 的实现，只是把选择队列部分换成上面的 sched_uid, sched_tid
        max_rate = 2442 * 1000 * 1000
        
        # 计算基础奖励（基于当前队列状态）
        max_vo, max_vi, max_be = self._compute_max_delays()
        
        # 1. 基础延迟成本（使用指数权重强调VO重要性）
        delay_cost = (
            self.cfg.w_vo * (max_vo ** 2) +  # VO延迟的平方，放大其影响
            self.cfg.w_vi * max_vi +
            self.cfg.w_be * max_be
        )
        
        # 2. 超时惩罚（对超过时延阈值的业务施加严厉惩罚）
        timeout_penalty = 0.0
        if max_vo > 20.0:  # VO超时阈值20ms
            timeout_penalty += 100.0 * (max_vo - 20.0)  # 每超1ms惩罚100
        if max_vi > 50.0:  # VI超时阈值50ms
            timeout_penalty += 20.0 * (max_vi - 50.0)   # 每超1ms惩罚20
        if max_be > 100.0: # BE超时阈值100ms
            timeout_penalty += 5.0 * (max_be - 100.0)   # 每超1ms惩罚5
        
        # 3. 综合reward
        total_cost = delay_cost + timeout_penalty
        total_cost = min(total_cost, 500.0)  # 裁剪上限，避免数值爆炸
        base_reward = -total_cost / 100.0
        
        # 4. 如果选择了空队列，施加额外惩罚
        if not self.use_traditional and selected_empty_queue:
            empty_queue_penalty = -5.0
            reward = base_reward + empty_queue_penalty
        else:
            reward = base_reward
        
        if sched_uid == -1:
            # 无包可发（所有队列都空，这是正常情况，不额外惩罚）
            heapq.heappush(self.event_queue, Task("schedule", self.current_time + 1.0))
            done = self.current_time > self.max_sim_time
            self.done = done
            return self._build_observation(), reward, done, {
                "sched_uid": sched_uid,
                "sched_tid": sched_tid,
                "time": self.current_time,
                "selected_empty_queue": selected_empty_queue if not self.use_traditional else False,
            }

        temp_sched_size = 0.0
        r_ptr = self.tx_info_table[sched_uid][sched_tid]["read_ptr"]
        w_ptr = self.tx_info_table[sched_uid][sched_tid]["write_ptr"]
        while r_ptr != w_ptr:
            if temp_sched_size == max_rate / 1000 * 4:
                break
            pkt_size = self.txq_table[sched_uid][sched_tid][r_ptr]["size"]
            if temp_sched_size + pkt_size <= max_rate / 1000 * 4:
                temp_sched_size += pkt_size
                r_ptr = (r_ptr + 1) % self.queue_size
            else:
                # 部分发送
                self.txq_table[sched_uid][sched_tid][r_ptr]["size"] = \
                    pkt_size - (max_rate / 1000 * 4 - temp_sched_size)
                temp_sched_size = max_rate / 1000 * 4

        self.tx_info_table[sched_uid][sched_tid]["read_ptr"] = r_ptr
        self.tx_info_table[sched_uid][sched_tid]["tot_size"] -= temp_sched_size

        send_time = temp_sched_size / max_rate * 1000.0
        if send_time <= 0:
            send_time = 0.1  # 防止除零

        heapq.heappush(self.event_queue, Task("schedule", self.current_time + send_time))

        # 4. 计算 reward（使用改进的超时惩罚机制）
        max_vo, max_vi, max_be = self._compute_max_delays()
        
        # 基础延迟成本（指数权重）
        delay_cost = (
            self.cfg.w_vo * (max_vo ** 2) +
            self.cfg.w_vi * max_vi +
            self.cfg.w_be * max_be
        )
        
        # 超时惩罚
        timeout_penalty = 0.0
        if max_vo > 20.0:
            timeout_penalty += 100.0 * (max_vo - 20.0)
        if max_vi > 50.0:
            timeout_penalty += 20.0 * (max_vi - 50.0)
        if max_be > 100.0:
            timeout_penalty += 5.0 * (max_be - 100.0)
        
        # 综合reward（原始值）
        total_cost = delay_cost + timeout_penalty
        total_cost = min(total_cost, 500.0)
        raw_reward = -total_cost / 100.0
        
        # 归一化reward（用于训练）
        # 使用最大时延作为当前流量难度的指示
        max_delay_overall = max(max_vo, max_vi, max_be)
        normalized_reward = self._normalize_reward(raw_reward, max_delay_overall)

        # 5. 终止判断
        done = self.current_time > self.max_sim_time
        self.done = done

        obs = self._build_observation()
        info = {
            "sched_uid": sched_uid,
            "sched_tid": sched_tid,
            "time": self.current_time,
            "max_vo_delay": max_vo,
            "max_vi_delay": max_vi,
            "max_be_delay": max_be,
            "selected_empty_queue": selected_empty_queue if not self.use_traditional else False,
            "raw_reward": raw_reward,  # 保存原始reward用于统计
        }
        
        # 记录到tracker
        if self.tracker is not None:
            queue_sizes, queue_delays = self._get_queue_snapshot()
            record = ScheduleRecord(
                timestamp=self.current_time,
                sched_uid=sched_uid,
                sched_tid=sched_tid,
                queue_sizes=queue_sizes,
                queue_delays=queue_delays,
                max_vo_delay=max_vo,
                max_vi_delay=max_vi,
                max_be_delay=max_be,
            )
            self.tracker.add_record(record)
        
        # 返回归一化的reward用于训练
        return obs, normalized_reward, done, info


class ActorCritic(nn.Module):
    """简单 MLP 的 Actor-Critic 网络"""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        hidden = 256
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value


def ppo_collect_rollout(env: RLSchedulingEnv,
                        model: nn.Module,
                        cfg: PPOConfig,
                        device: torch.device):
    """
    收集一批 rollout 数据（单环境版本，方便理解；需要多核时可扩展为向量环境）
    返回值增加 raw_rew_t 用于真实的reward统计
    """
    obs_buf, act_buf, logp_buf, rew_buf, raw_rew_buf, val_buf, done_buf = [], [], [], [], [], [], []
    obs = env.reset()
    for _ in range(cfg.rollout_steps):
        obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        next_obs, reward, done, info = env.step(action.item())

        obs_buf.append(obs)
        act_buf.append(action.cpu().numpy())
        logp_buf.append(logp.cpu().numpy())
        rew_buf.append(reward)  # 归一化后的reward用于训练
        raw_rew_buf.append(info.get('raw_reward', reward))  # 原始reward用于统计
        val_buf.append(value.cpu().numpy())
        done_buf.append(done)

        obs = next_obs
        if done:
            obs = env.reset()

    # 转成 tensor
    obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
    act_t = torch.tensor(np.array(act_buf).squeeze(-1), dtype=torch.long, device=device)
    logp_t = torch.tensor(np.array(logp_buf).squeeze(-1), dtype=torch.float32, device=device)
    rew_t = torch.tensor(np.array(rew_buf), dtype=torch.float32, device=device)
    raw_rew_t = torch.tensor(np.array(raw_rew_buf), dtype=torch.float32, device=device)
    val_t = torch.tensor(np.array(val_buf), dtype=torch.float32, device=device)
    done_t = torch.tensor(np.array(done_buf), dtype=torch.float32, device=device)

    # GAE-Lambda 计算 advantage（使用归一化后的reward）
    adv_buf = torch.zeros_like(rew_t, device=device)
    ret_buf = torch.zeros_like(rew_t, device=device)
    last_adv = 0.0
    last_ret = 0.0
    for t in reversed(range(cfg.rollout_steps)):
        mask = 1.0 - done_t[t]
        delta = rew_t[t] + cfg.gamma * (val_t[t + 1] if t + 1 < cfg.rollout_steps else 0.0) * mask - val_t[t]
        last_adv = delta + cfg.gamma * cfg.lam * mask * last_adv
        adv_buf[t] = last_adv
        last_ret = val_t[t] + last_adv
        ret_buf[t] = last_ret

    # 保存未归一化的advantage和return用于统计分析
    raw_adv_buf = adv_buf.clone()
    raw_ret_buf = ret_buf.clone()
    
    # 对advantage进行归一化（用于训练）
    adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

    return obs_t, act_t, logp_t, adv_buf, ret_buf, raw_rew_t, raw_adv_buf, raw_ret_buf  # 返回原始数据用于统计


def ppo_update(model: nn.Module,
               optimizer: optim.Optimizer,
               cfg: PPOConfig,
               batch_obs, batch_act, batch_logp_old, batch_adv, batch_ret,
               device: torch.device):
    N = batch_obs.size(0)
    idxs = np.arange(N)
    for _ in range(cfg.epochs):
        np.random.shuffle(idxs)
        for start in range(0, N, cfg.minibatch_size):
            end = start + cfg.minibatch_size
            mb_idx = idxs[start:end]

            obs = batch_obs[mb_idx]
            act = batch_act[mb_idx]
            logp_old = batch_logp_old[mb_idx]
            adv = batch_adv[mb_idx]
            ret = batch_ret[mb_idx]

            logits, value = model(obs)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - logp_old)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (ret - value).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()


def train_ppo(cfg: PPOConfig):
    """主训练入口：训练 PPO 并画出奖励收敛图"""
    device = torch.device(cfg.device)
    set_global_seeds(cfg.seed)

    # 环境 & 模型
    # 创建训练环境（启用reward归一化）
    env = RLSchedulingEnv(cfg, use_traditional=False, normalize_reward=True)
    model = ActorCritic(env.obs_dim, env.act_dim).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    episode_rewards = []  # 训练过程中的reward（不同流量，不可直接比较）
    eval_rewards = []  # 固定评估集的reward（可比较）
    eval_improvement = []  # 相对于baseline的改善率
    advantage_stats = []  # 记录每个episode的advantage统计信息（归一化后）
    raw_advantage_stats = []  # 记录每个episode的原始advantage统计信息（未归一化）
    raw_return_stats = []  # 记录每个episode的原始return统计信息（未归一化）
    
    # 固定评估集：每10个episode在相同流量下评估
    eval_frequency = 10
    eval_seeds = [1000, 1001, 1002]  # 3个固定的流量场景
    
    # 计算baseline（传统策略）在评估集上的性能作为参考
    print("\n[评估] 计算baseline性能...")
    baseline_rewards = []
    for eval_seed in eval_seeds:
        base_rew = evaluate_policy(cfg, model=None, use_traditional=True, 
                                   seed=eval_seed, enable_tracking=False)
        baseline_rewards.append(base_rew)
    baseline_avg = np.mean(baseline_rewards)
    print(f"[评估] Baseline平均reward: {baseline_avg:.6f}")

    # 使用 tqdm 展示 episode 进度
    for ep in tqdm(range(cfg.total_episodes), desc="PPO Training"):
        # 收集一批 rollout（随机流量训练）
        obs_t, act_t, logp_t, adv_t, ret_t, raw_rew_t, raw_adv_t, raw_ret_t = ppo_collect_rollout(env, model, cfg, device)

        # 训练过程的reward（仅供参考，因为流量每次不同）
        with torch.no_grad():
            avg_ep_rew = raw_rew_t.mean().item()
            
            # 记录归一化后的advantage统计信息（训练质量指标）
            adv_mean = adv_t.mean().item()
            adv_std = adv_t.std().item()
            adv_max = adv_t.max().item()
            adv_min = adv_t.min().item()
            
            # 记录未归一化的advantage统计信息（真实的价值估计）
            raw_adv_mean = raw_adv_t.mean().item()
            raw_adv_std = raw_adv_t.std().item()
            raw_adv_max = raw_adv_t.max().item()
            raw_adv_min = raw_adv_t.min().item()
            
            # 记录未归一化的return统计信息（真实的累积回报）
            raw_ret_mean = raw_ret_t.mean().item()
            raw_ret_std = raw_ret_t.std().item()
            raw_ret_max = raw_ret_t.max().item()
            raw_ret_min = raw_ret_t.min().item()
        
        episode_rewards.append(avg_ep_rew)
        advantage_stats.append({
            'mean': adv_mean,
            'std': adv_std,
            'max': adv_max,
            'min': adv_min
        })
        raw_advantage_stats.append({
            'mean': raw_adv_mean,
            'std': raw_adv_std,
            'max': raw_adv_max,
            'min': raw_adv_min
        })
        raw_return_stats.append({
            'mean': raw_ret_mean,
            'std': raw_ret_std,
            'max': raw_ret_max,
            'min': raw_ret_min
        })

        # PPO 更新
        ppo_update(model, optimizer, cfg, obs_t, act_t, logp_t, adv_t, ret_t, device)
        
        # 定期在固定评估集上测试（可比较的指标）
        if (ep + 1) % eval_frequency == 0 or ep == 0:
            eval_rews = []
            for eval_seed in eval_seeds:
                eval_rew = evaluate_policy(cfg, model, use_traditional=False, 
                                           seed=eval_seed, enable_tracking=False)
                eval_rews.append(eval_rew)
            eval_avg = np.mean(eval_rews)
            eval_rewards.append(eval_avg)
            
            # 计算相对于baseline的改善率
            improvement = ((eval_avg - baseline_avg) / abs(baseline_avg)) * 100
            eval_improvement.append(improvement)
            
            tqdm.write(f"Ep {ep+1}: 评估reward={eval_avg:.6f}, 相对baseline改善={improvement:+.2f}%")

    # 画奖励收敛图（折线图）
    os.makedirs("results", exist_ok=True)
    
    # 图1：Reward曲线（分两个子图）
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 子图1：训练过程reward（随机流量，仅供参考）
    axes[0].plot(episode_rewards, linewidth=1, alpha=0.5, label="Training reward (random traffic)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title("Training Reward (not directly comparable due to random traffic)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].text(0.02, 0.98, "⚠️ 注意：每个episode流量不同，此曲线不能直接比较",
                 transform=axes[0].transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # 子图2：固定评估集reward（可比较）
    eval_episodes = [i * eval_frequency for i in range(len(eval_rewards))]
    if len(eval_episodes) > 0 and eval_episodes[0] != 0:
        eval_episodes = [0] + eval_episodes
    axes[1].plot(eval_episodes, eval_rewards, linewidth=2, marker='o', markersize=6,
                 label="Evaluation reward (fixed traffic)", color='green')
    axes[1].axhline(y=baseline_avg, color='red', linestyle='--', linewidth=2,
                    label=f'Baseline (traditional): {baseline_avg:.6f}', alpha=0.7)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Average Reward")
    axes[1].set_title("Evaluation Reward on Fixed Traffic Scenarios (comparable)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].text(0.02, 0.98, "✓ 相同流量场景，可以直接比较",
                 transform=axes[1].transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    fig_path = os.path.join("results", "ppo_reward_curve.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[PPO] 收敛曲线已保存到 {fig_path}")
    
    # 图1.5：相对改善率曲线
    plt.figure(figsize=(12, 6))
    plt.plot(eval_episodes, eval_improvement, linewidth=2, marker='s', markersize=6,
             color='blue', label='Improvement over baseline')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline (0%)')
    plt.xlabel("Episode")
    plt.ylabel("Improvement (%)")
    plt.title("PPO Performance Improvement over Traditional Scheduling")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join("results", "ppo_improvement_curve.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[PPO] 改善率曲线已保存到 {fig_path}")
    
    # 图2：Advantage统计（训练质量指标）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 提取advantage统计数据
    adv_means = [s['mean'] for s in advantage_stats]
    adv_stds = [s['std'] for s in advantage_stats]
    adv_maxs = [s['max'] for s in advantage_stats]
    adv_mins = [s['min'] for s in advantage_stats]
    episodes = list(range(len(advantage_stats)))
    
    # 子图1：Advantage均值（应该接近0）
    axes[0, 0].plot(episodes, adv_means, linewidth=2, color='blue')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Target (0)')
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Advantage Mean")
    axes[0, 0].set_title("Advantage Mean (should be near 0)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 子图2：Advantage标准差（反映策略稳定性）
    axes[0, 1].plot(episodes, adv_stds, linewidth=2, color='green')
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Advantage Std")
    axes[0, 1].set_title("Advantage Std (training stability indicator)")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3：Advantage范围（max和min）
    axes[1, 0].plot(episodes, adv_maxs, linewidth=2, color='red', label='Max', alpha=0.7)
    axes[1, 0].plot(episodes, adv_mins, linewidth=2, color='blue', label='Min', alpha=0.7)
    axes[1, 0].fill_between(episodes, adv_mins, adv_maxs, alpha=0.2)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Advantage Value")
    axes[1, 0].set_title("Advantage Range (Max & Min)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 子图4：Reward和Advantage Std的关系（双Y轴）
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(episodes, episode_rewards, linewidth=2, color='purple', label='Reward', alpha=0.8)
    line2 = ax2.plot(episodes, adv_stds, linewidth=2, color='orange', label='Adv Std', alpha=0.8)
    
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Average Reward", color='purple')
    ax2.set_ylabel("Advantage Std", color='orange')
    ax1.tick_params(axis='y', labelcolor='purple')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax1.set_title("Reward vs Advantage Std")
    ax1.grid(True, alpha=0.3)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    
    plt.tight_layout()
    fig_path = os.path.join("results", "ppo_advantage_analysis.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[PPO] Advantage分析图已保存到 {fig_path}")
    
    # 图3：新增综合分析图 - 固定流量评估 + 原始Advantage统计（未归一化）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 提取原始advantage和return统计数据
    raw_adv_means = [s['mean'] for s in raw_advantage_stats]
    raw_adv_stds = [s['std'] for s in raw_advantage_stats]
    raw_adv_maxs = [s['max'] for s in raw_advantage_stats]
    raw_adv_mins = [s['min'] for s in raw_advantage_stats]
    
    raw_ret_means = [s['mean'] for s in raw_return_stats]
    raw_ret_stds = [s['std'] for s in raw_return_stats]
    raw_ret_maxs = [s['max'] for s in raw_return_stats]
    raw_ret_mins = [s['min'] for s in raw_return_stats]
    
    # 子图1：固定评估集Reward（真正的收敛指标）
    ax = axes[0, 0]
    eval_episodes = [i * eval_frequency for i in range(len(eval_rewards))]
    if len(eval_episodes) > 0 and eval_episodes[0] != 0:
        eval_episodes = [0] + eval_episodes
    ax.plot(eval_episodes, eval_rewards, linewidth=2.5, marker='o', markersize=8,
            label="PPO策略 (固定流量评估)", color='green', markerfacecolor='white', 
            markeredgewidth=2)
    ax.axhline(y=baseline_avg, color='red', linestyle='--', linewidth=2.5,
               label=f'Baseline: {baseline_avg:.6f}', alpha=0.8)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Average Reward", fontsize=12)
    ax.set_title("固定流量评估Reward（真实收敛曲线）", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    ax.text(0.02, 0.02, "✓ 这条线能反映真实的策略改进", 
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 子图2：原始Advantage均值趋势（未归一化）
    ax = axes[0, 1]
    ax.plot(episodes, raw_adv_means, linewidth=2, color='darkblue', alpha=0.8)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Raw Advantage Mean", fontsize=12)
    ax.set_title("原始Advantage均值（未归一化）", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, "说明：这是归一化前的真实advantage值\n如果策略改善，期望看到上升趋势", 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # 子图3：原始Return均值趋势（未归一化）
    ax = axes[1, 0]
    ax.plot(episodes, raw_ret_means, linewidth=2, color='purple', alpha=0.8)
    # 添加移动平均线以显示趋势
    if len(raw_ret_means) >= 10:
        window = 10
        smoothed = np.convolve(raw_ret_means, np.ones(window)/window, mode='valid')
        smooth_episodes = episodes[window-1:]
        ax.plot(smooth_episodes, smoothed, linewidth=2.5, color='red', 
                label=f'{window}-episode moving average', alpha=0.9)
        ax.legend(fontsize=10)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Raw Return Mean", fontsize=12)
    ax.set_title("原始Return均值（未归一化）", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, "说明：累积折扣回报的真实值\n上升趋势说明策略在改善", 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
    
    # 子图4：原始Advantage标准差（反映价值估计的不确定性）
    ax = axes[1, 1]
    ax.plot(episodes, raw_adv_stds, linewidth=2, color='orange', alpha=0.8)
    # 添加移动平均线
    if len(raw_adv_stds) >= 10:
        window = 10
        smoothed = np.convolve(raw_adv_stds, np.ones(window)/window, mode='valid')
        smooth_episodes = episodes[window-1:]
        ax.plot(smooth_episodes, smoothed, linewidth=2.5, color='darkred', 
                label=f'{window}-episode moving average', alpha=0.9)
        ax.legend(fontsize=10)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Raw Advantage Std", fontsize=12)
    ax.set_title("原始Advantage标准差（未归一化）", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, "说明：价值估计的波动幅度\n趋于稳定说明策略和价值估计收敛", 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.5))
    
    plt.tight_layout()
    fig_path = os.path.join("results", "ppo_convergence_comprehensive.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[PPO] 综合收敛分析图已保存到 {fig_path}")
    
    # 生成训练诊断报告
    print("\n" + "="*70)
    print("📊 PPO训练诊断报告")
    print("="*70)
    
    # 1. 固定评估集性能分析（最重要！）
    if len(eval_rewards) > 1:
        eval_improvement_value = eval_rewards[-1] - eval_rewards[0]
        final_improvement_pct = eval_improvement[-1] if len(eval_improvement) > 0 else 0
        print(f"\n【固定评估集性能】⭐ (可比较的真实指标)")
        print(f"  Baseline (传统调度): {baseline_avg:.6f}")
        print(f"  初始PPO性能: {eval_rewards[0]:.6f}")
        print(f"  最终PPO性能: {eval_rewards[-1]:.6f}")
        print(f"  绝对改善: {eval_improvement_value:.6f}")
        print(f"  相对baseline改善: {final_improvement_pct:+.2f}%")
        if final_improvement_pct > 5:
            print(f"  ✓ PPO显著优于传统调度")
        elif final_improvement_pct > 0:
            print(f"  ⚠ PPO略优于传统调度")
        else:
            print(f"  ✗ PPO未超过传统调度")
    
    # 2. 训练过程Reward分析（仅供参考）
    reward_improvement = episode_rewards[-1] - episode_rewards[0] if len(episode_rewards) > 0 else 0
    print(f"\n【训练过程Reward】(随机流量，仅供参考)")
    print(f"  初始: {episode_rewards[0]:.6f}")
    print(f"  最终: {episode_rewards[-1]:.6f}")
    print(f"  变化: {reward_improvement:.6f}")
    print(f"  ⚠ 注意：每个episode流量不同，此数值不能直接比较")
    
    # 2. Advantage稳定性分析
    first_10_std = np.mean([s['std'] for s in advantage_stats[:10]])
    last_10_std = np.mean([s['std'] for s in advantage_stats[-10:]])
    std_change = last_10_std - first_10_std
    stability_trend = "改善✓" if std_change < 0 else "未改善✗"
    
    print(f"\n【Advantage稳定性分析】")
    print(f"  前10轮平均Std: {first_10_std:.6f}")
    print(f"  后10轮平均Std: {last_10_std:.6f}")
    print(f"  变化: {std_change:.6f} ({stability_trend})")
    
    # 3. Advantage范围分析
    first_10_range = np.mean([s['max'] - s['min'] for s in advantage_stats[:10]])
    last_10_range = np.mean([s['max'] - s['min'] for s in advantage_stats[-10:]])
    range_change = last_10_range - first_10_range
    range_trend = "收窄✓" if range_change < 0 else "未收窄✗"
    
    print(f"\n【Advantage范围分析】")
    print(f"  前10轮平均范围: {first_10_range:.4f}")
    print(f"  后10轮平均范围: {last_10_range:.4f}")
    print(f"  变化: {range_change:.4f} ({range_trend})")
    
    # 3.5 原始Advantage和Return分析（未归一化）
    first_10_raw_adv = np.mean([s['mean'] for s in raw_advantage_stats[:10]])
    last_10_raw_adv = np.mean([s['mean'] for s in raw_advantage_stats[-10:]])
    raw_adv_change = last_10_raw_adv - first_10_raw_adv
    
    first_10_raw_ret = np.mean([s['mean'] for s in raw_return_stats[:10]])
    last_10_raw_ret = np.mean([s['mean'] for s in raw_return_stats[-10:]])
    raw_ret_change = last_10_raw_ret - first_10_raw_ret
    
    print(f"\n【原始价值估计分析】(未归一化)")
    print(f"  原始Advantage均值变化:")
    print(f"    前10轮: {first_10_raw_adv:.6f}")
    print(f"    后10轮: {last_10_raw_adv:.6f}")
    print(f"    变化: {raw_adv_change:+.6f} {'↑' if raw_adv_change > 0 else '↓'}")
    print(f"  原始Return均值变化:")
    print(f"    前10轮: {first_10_raw_ret:.6f}")
    print(f"    后10轮: {last_10_raw_ret:.6f}")
    print(f"    变化: {raw_ret_change:+.6f} {'↑' if raw_ret_change > 0 else '↓'}")
    if raw_adv_change > 0 and raw_ret_change > 0:
        print(f"  ✓ 价值估计显示策略改善趋势")
    elif raw_adv_change > 0 or raw_ret_change > 0:
        print(f"  ⚠ 价值估计显示部分改善")
    else:
        print(f"  ✗ 价值估计未显示明显改善")
    
    # 4. 训练质量总结
    print(f"\n【训练质量总结】")
    
    # 基于固定评估集判断
    if len(eval_rewards) > 1:
        if final_improvement_pct > 5:
            print(f"  ✓ 策略性能显著改善（相对baseline +{final_improvement_pct:.1f}%）")
        elif final_improvement_pct > 0:
            print(f"  ⚠ 策略性能轻微改善（相对baseline +{final_improvement_pct:.1f}%）")
        else:
            print(f"  ✗ 策略未超过baseline")
    
    if std_change < -0.01:
        print(f"  ✓ 策略趋于稳定")
    elif abs(std_change) < 0.01:
        print(f"  ⚠ 策略稳定性无明显变化")
    else:
        print(f"  ✗ 策略不稳定")
    
    if range_change < -0.5:
        print(f"  ✓ 动作质量趋于一致")
    elif abs(range_change) < 0.5:
        print(f"  ⚠ 动作质量无明显变化")
    else:
        print(f"  ✗ 动作质量参差不齐")
    
    # 5. 改进建议
    print(f"\n【改进建议】")
    if len(eval_rewards) > 1 and final_improvement_pct < 5:
        print(f"  • 增加训练轮数 (当前: {cfg.total_episodes}, 建议: {cfg.total_episodes*5})")
        print(f"  • 调整奖励函数权重 (VO:{cfg.w_vo}, VI:{cfg.w_vi}, BE:{cfg.w_be})")
    if std_change >= 0:
        print(f"  • 降低学习率 (当前: {cfg.lr}, 建议: {cfg.lr/2:.2e})")
        print(f"  • 减小clip范围 (当前: {cfg.clip_eps}, 建议: 0.1)")
    if abs(std_change) < 0.01:
        print(f"  • 增加熵系数鼓励探索 (建议: 0.05)")
    
    print("="*70 + "\n")

    return model, episode_rewards


def evaluate_policy(env_cfg: PPOConfig, model: nn.Module = None, use_traditional: bool = False, 
                   seed: int = 1234, enable_tracking: bool = False):
    """
    对策略进行评估：
    - use_traditional=True：传统调度
    - use_traditional=False 且提供 model：用 PPO 策略
    - enable_tracking=True：返回详细的调度追踪数据
    
    Returns:
        如果 enable_tracking=False: 返回 total_reward（原始reward）
        如果 enable_tracking=True: 返回 (total_reward, tracker)
    """
    device = torch.device(env_cfg.device)
    set_global_seeds(seed)
    # 评估时不使用reward归一化，直接用原始reward比较
    env = RLSchedulingEnv(env_cfg, use_traditional=use_traditional, 
                         enable_tracking=enable_tracking, normalize_reward=False)
    obs = env.reset(seed=seed)
    total_reward = 0.0

    while True:
        if use_traditional:
            # 动作无意义，内部会走 gen_sched_res
            action = 0
        else:
            obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
            with torch.no_grad():
                logits, value = model(obs_t)
                # 评估时用贪心动作，保证确定性
                action = torch.argmax(logits, dim=-1).item()

        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    if enable_tracking:
        return total_reward, env.tracker
    else:
        return total_reward


def visualize_comparison(tracker_traditional: ScheduleTracker, 
                        tracker_ppo: ScheduleTracker,
                        save_dir: str = "results"):
    """
    可视化对比两种调度策略
    
    生成多个对比图：
    1. 时间-各业务类型最大时延对比
    2. 时间-总队列流量对比  
    3. 调度选择分布对比
    4. 特定用户队列流量对比（示例：user 0 的三个队列）
    5. 时延热力图对比
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取数据
    times_trad = tracker_traditional.get_timestamps()
    times_ppo = tracker_ppo.get_timestamps()
    
    vo_trad, vi_trad, be_trad = tracker_traditional.get_max_delays()
    vo_ppo, vi_ppo, be_ppo = tracker_ppo.get_max_delays()
    
    total_queue_trad = tracker_traditional.get_total_queue_sizes()
    total_queue_ppo = tracker_ppo.get_total_queue_sizes()
    
    # === 图1: 时间-最大时延对比（三个子图） ===
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axes[0].plot(times_trad, vo_trad, label='Traditional', alpha=0.7, linewidth=1)
    axes[0].plot(times_ppo, vo_ppo, label='PPO', alpha=0.7, linewidth=1)
    axes[0].set_ylabel('VO Max Delay (ms)')
    axes[0].set_title('VO (Voice) Traffic Maximum Delay Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(times_trad, vi_trad, label='Traditional', alpha=0.7, linewidth=1)
    axes[1].plot(times_ppo, vi_ppo, label='PPO', alpha=0.7, linewidth=1)
    axes[1].set_ylabel('VI Max Delay (ms)')
    axes[1].set_title('VI (Video) Traffic Maximum Delay Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(times_trad, be_trad, label='Traditional', alpha=0.7, linewidth=1)
    axes[2].plot(times_ppo, be_ppo, label='PPO', alpha=0.7, linewidth=1)
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('BE Max Delay (ms)')
    axes[2].set_title('BE (Best Effort) Traffic Maximum Delay Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "delay_comparison.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[Visualization] 时延对比图已保存到 {fig_path}")
    
    # === 图2: 时间-总队列流量对比 ===
    plt.figure(figsize=(12, 6))
    plt.plot(times_trad, total_queue_trad, label='Traditional', alpha=0.7, linewidth=1)
    plt.plot(times_ppo, total_queue_ppo, label='PPO', alpha=0.7, linewidth=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Total Queue Size (bytes)')
    plt.title('Total Queue Size Comparison Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "queue_size_comparison.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[Visualization] 队列流量对比图已保存到 {fig_path}")
    
    # === 图3: 调度选择分布对比（折线图）===
    summary_trad = tracker_traditional.summary()
    summary_ppo = tracker_ppo.summary()
    
    dist_trad = summary_trad['schedule_distribution']
    dist_ppo = summary_ppo['schedule_distribution']
    
    # 确保两个字典有相同的键
    all_keys = sorted(set(dist_trad.keys()) | set(dist_ppo.keys()))
    counts_trad = [dist_trad.get(k, 0) for k in all_keys]
    counts_ppo = [dist_ppo.get(k, 0) for k in all_keys]
    
    x = np.arange(len(all_keys))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, counts_trad, marker='o', markersize=8, linewidth=2, label='Traditional', alpha=0.8)
    ax.plot(x, counts_ppo, marker='s', markersize=8, linewidth=2, label='PPO', alpha=0.8)
    
    ax.set_xlabel('Access Category')
    ax.set_ylabel('Scheduling Count')
    ax.set_title('Scheduling Decision Distribution Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(all_keys)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (trad_val, ppo_val) in enumerate(zip(counts_trad, counts_ppo)):
        if trad_val > 0:
            ax.text(i, trad_val, f'{int(trad_val)}', ha='center', va='bottom', fontsize=8)
        if ppo_val > 0:
            ax.text(i, ppo_val, f'{int(ppo_val)}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "schedule_distribution_comparison.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[Visualization] 调度分布对比图已保存到 {fig_path}")
    
    # === 图4: 特定用户队列流量对比（示例：user 0）===
    user_id = 0
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for tid in range(3):
        queue_trad = tracker_traditional.get_queue_size_by_user_tid(user_id, tid)
        queue_ppo = tracker_ppo.get_queue_size_by_user_tid(user_id, tid)
        
        axes[tid].plot(times_trad, queue_trad, label='Traditional', alpha=0.7, linewidth=1)
        axes[tid].plot(times_ppo, queue_ppo, label='PPO', alpha=0.7, linewidth=1)
        axes[tid].set_ylabel(f'Queue Size (bytes)')
        ac_names = ['VO', 'VI', 'BE']
        axes[tid].set_title(f'User {user_id} - {ac_names[tid]} Queue Size Over Time')
        axes[tid].legend()
        axes[tid].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Time (ms)')
    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"user{user_id}_queue_comparison.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[Visualization] 用户{user_id}队列对比图已保存到 {fig_path}")
    
    # === 图5: 平均时延对比摘要（折线图）===
    metrics = ['VO Avg Delay', 'VI Avg Delay', 'BE Avg Delay']
    trad_vals = [summary_trad['avg_vo_delay'], summary_trad['avg_vi_delay'], summary_trad['avg_be_delay']]
    ppo_vals = [summary_ppo['avg_vo_delay'], summary_ppo['avg_vi_delay'], summary_ppo['avg_be_delay']]
    
    x = np.arange(len(metrics))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, trad_vals, marker='o', markersize=8, linewidth=2, label='Traditional', alpha=0.8)
    ax.plot(x, ppo_vals, marker='s', markersize=8, linewidth=2, label='PPO', alpha=0.8)
    
    ax.set_ylabel('Average Delay (ms)')
    ax.set_xlabel('Traffic Type')
    ax.set_title('Average Delay Comparison by Traffic Type')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (trad_val, ppo_val) in enumerate(zip(trad_vals, ppo_vals)):
        ax.text(i, trad_val, f'{trad_val:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(i, ppo_val, f'{ppo_val:.2f}', ha='center', va='top', fontsize=9)
    
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "avg_delay_comparison.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[Visualization] 平均时延对比图已保存到 {fig_path}")
    
    # === 打印统计摘要 ===
    print("\n" + "="*60)
    print("统计摘要对比")
    print("="*60)
    print(f"\n【传统调度】")
    for key, val in summary_trad.items():
        if key != 'schedule_distribution':
            print(f"  {key}: {val}")
    print(f"  调度分布: {summary_trad['schedule_distribution']}")
    
    print(f"\n【PPO调度】")
    for key, val in summary_ppo.items():
        if key != 'schedule_distribution':
            print(f"  {key}: {val}")
    print(f"  调度分布: {summary_ppo['schedule_distribution']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    """
    运行方式示例（命令行）：
    1. 仅跑原始传统调度仿真（不训练）：
       python simcore.py --mode baseline

    2. 训练 PPO：
       python simcore.py --mode train_ppo

    3. 训练后对比传统调度与 PPO（使用相同流量随机序列）：
       python simcore.py --mode compare
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str,
                        choices=["baseline", "train_ppo", "compare"],
                        default="train_ppo")
    args = parser.parse_args()

    if args.mode == "baseline":
        # 原始传统调度仿真（保持你的原逻辑，只是打包成函数）
        user_num = 32
        simu_time = 10000  # 毫秒为单位
        queue_size = 512   # 使用原来的队列大小
        txq_table = create_txq_table(user_num, queue_size)
        tx_info_table = create_tx_info_table(user_num)

        priority_queue = []
        heapq.heappush(priority_queue, Task("traffic_generate", 0.0))
        heapq.heappush(priority_queue, Task("schedule", 0.5))
        duration_per_time = 10000
        runtimes = 100

        rate_table = create_rate_table(user_num)
        while True:
            task = heapq.heappop(priority_queue)
            if task.event == "traffic_generate":
                traffic_generator(
                    txq_table, user_num, task.timestamp,
                    rate_table, tx_info_table, queue_size,
                    priority_queue
                )
            else:
                schedule(
                    txq_table, tx_info_table, user_num,
                    task.timestamp, queue_size, priority_queue
                )

            if math.floor(task.timestamp) % duration_per_time == 0:
                rate_table = create_rate_table(user_num)
            if task.timestamp > runtimes * duration_per_time:
                break

    elif args.mode == "train_ppo":
        cfg = PPOConfig()
        model, rewards = train_ppo(cfg)

    elif args.mode == "compare":
        # 对比传统调度和 PPO 策略，保证使用相同流量随机性
        cfg = PPOConfig()
        # 先训练一个模型（也可以改成直接 load 已训练好的参数）
        print("[对比模式] 开始训练PPO模型...")
        model, rewards = train_ppo(cfg)

        seed = 2024
        print(f"\n[对比模式] 使用种子 {seed} 进行评估对比...")
        print("[对比模式] 评估传统调度...")
        base_ret, tracker_trad = evaluate_policy(cfg, model=None, use_traditional=True, 
                                                  seed=seed, enable_tracking=True)
        print("[对比模式] 评估PPO调度...")
        ppo_ret, tracker_ppo = evaluate_policy(cfg, model=model, use_traditional=False, 
                                                seed=seed, enable_tracking=True)

        print(f"\n传统调度 总回报: {base_ret:.4f}")
        print(f"PPO 调度 总回报: {ppo_ret:.4f}")
        
        # 生成对比可视化
        print("\n[对比模式] 生成可视化对比图...")
        visualize_comparison(tracker_trad, tracker_ppo, save_dir="results")
        print("\n[对比模式] 完成！所有对比图已保存到 results/ 目录")
