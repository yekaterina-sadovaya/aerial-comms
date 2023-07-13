import numpy as np
import math
import random
import copy
import heapq


class AerDeployment:
    def __init__(self):
        self.n_ues = 10
        self.n_bs = 2
        self.n_uav = 2
        self.R_haps = 1500
        self.rate_parameter = 1/20
        self.n_tasks = int(1e2)
        self.precision = 3
        self.task_size = 5
        self.seed = 1
        self.fc_haps = 60e9
        self.fc_bs = 30e9
        self.fc_uav = 40e9
        self.tx_pow_ue = 15
        self.tx_pow_bs = 25
        self.tx_pow_uav = 20
        self.tx_pow_haps = 40
        # nodes are enumerated in the following way 0-n_ue (ues), n_ue-n_ue+n_bs (bss),
        # n_ue+n_bs-n_ue+n_bs+n_uav (uavs), n_ue+n_bs+n_uav-n_ue+n_bs+n_uav+1 (haps)
        self.channel_matrix = np.zeros([self.n_ues + self.n_bs + self.n_uav + 1,
                                        self.n_ues + self.n_bs + self.n_uav + 1])
        self.propagation_dists = np.zeros([self.n_ues + self.n_bs + self.n_uav + 1,
                                           self.n_ues + self.n_bs + self.n_uav + 1])
        self.mc_degree = 2
        self.active_connections = np.zeros([self.n_ues, self.mc_degree])
        self.haps_height = 1500
        self.uav_height = 200
        self.bs_height = 20
        self.ue_height = 1.5
        self.ue_pos = np.zeros([self.n_ues, 3])
        self.bs_pos = np.zeros([self.n_bs, 3])
        self.haps_pos = np.array([0, 0, self.haps_height])
        self.uav_pos = np.zeros([self.n_uav, 3])
        self.thermal_noise_density = -173.93

    def drop_in_circle(self, n_points, height):
        random.seed(self.seed)
        R = self.R_haps
        pos = np.zeros([n_points, 3])
        for n in range(n_points):
            r = R * math.sqrt(random.random())
            theta = random.random() * 2 * math.pi
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            pos[n, :] = [x, y, height]
        return pos

    def drop_nodes(self):
        self.ue_pos = self.drop_in_circle(self.n_ues, self.ue_height)
        self.bs_pos = self.drop_in_circle(self.n_bs, self.bs_height)
        self.uav_pos = self.drop_in_circle(self.n_uav, self.uav_height)

    def UMa_LoS(self, dist, f_carrier, h_bs=25.0, h_ms=1.5):
        d_bp = 4.0 * (h_bs - 1.0) * (h_ms - 1.0) * f_carrier / 3.0e8
        PL_MAX = 200
        if dist > 5000.0:
            pl = PL_MAX
        else:
            if dist < 10.0:
                dist = 10.0 + (10.0 - dist)

            if dist < d_bp:
                pl = 22 * math.log10(dist) + 28 + 20 * math.log10(f_carrier / 1e9)
            else:
                pl = 40.0 * math.log10(dist) + 7.8 - 18.0 * math.log10(h_bs - 1) \
                     - 18.0 * math.log10(h_ms - 1) + 2.0 * math.log10(f_carrier / 1e9)
        return pl

    def compute_channels(self):
        for link1_num in range(self.n_ues + self.n_bs + self.n_uav + 1):
            for link2_num in range(self.n_ues + self.n_bs + self.n_uav + 1):
                if (link1_num < self.n_ues) and (link2_num < self.n_ues):
                    continue
                else:
                    if link1_num < self.n_ues:
                        tx_pow = self.tx_pow_ue
                        height1 = self.ue_height
                        pos1 = self.ue_pos[link1_num, :]
                    elif self.n_ues <= link1_num < self.n_ues + self.n_bs:
                        tx_pow = self.tx_pow_bs
                        height1 = self.bs_height
                        pos1 = self.ue_pos[self.n_bs - (link1_num - self.n_ues), :]
                    elif self.n_ues + self.n_bs <= link1_num < self.n_ues + self.n_bs + self.n_uav:
                        tx_pow = self.tx_pow_uav
                        height1 = self.uav_height
                        pos1 = self.ue_pos[self.n_uav - (link1_num - self.n_ues - self.n_bs), :]
                    else:
                        tx_pow = self.tx_pow_haps
                        height1 = self.haps_height
                        pos1 = self.haps_pos

                    if link2_num < self.n_ues:
                        height2 = self.ue_height
                        pos2 = self.ue_pos[link2_num, :]
                        fc = self.fc_uav
                    elif self.n_ues <= link2_num < self.n_ues + self.n_bs:
                        height2 = self.bs_height
                        pos2 = self.bs_pos[link2_num - self.n_ues, :]
                        fc = self.fc_bs
                    elif self.n_ues + self.n_bs <= link2_num < self.n_ues + self.n_bs + self.n_uav:
                        height2 = self.uav_height
                        pos2 = self.uav_pos[link2_num - self.n_ues - self.n_bs, :]
                        fc = self.fc_uav
                    else:
                        height2 = self.haps_height
                        pos2 = self.haps_pos
                        fc = self.fc_haps

                    dist = np.linalg.norm(pos1 - pos2)
                    rx_pow = tx_pow - self.UMa_LoS(dist, fc, height1, height2) - self.thermal_noise_density
                    self.channel_matrix[link1_num, link2_num] = 10.0 ** ((rx_pow-30) / 10.0)
                    self.propagation_dists[link1_num, link2_num] = dist

        for ue in range(self.n_ues):
            ch = copy.copy(self.channel_matrix[ue, :])
            for n in range(self.mc_degree):
                max_num = max(ch)
                max_id = np.where(ch == max_num)[0][0]
                ch[max_id] = 0
                self.active_connections[ue, n] = max_id


class Task:
    def __init__(self, arrival_time, user_id):
        self.arrival_time = arrival_time
        self.user_id = user_id
        self.queue_entry_time = 0


class Server:
    def __init__(self, processing_time, server_id, comp_resources):
        self.processing_time = processing_time
        self.server_id = server_id
        self.tasks_queue = []
        self.comp_resources = comp_resources
        self.current_time = 0


class BasicOffloading:
    def __init__(self, ue_connections, delay_matrix, num_tasks, mean_arrival_rate, n_bs, n_uav):
        self.num_tasks = num_tasks
        self.dist_matrix = delay_matrix
        self.mean_arrival_rate = mean_arrival_rate
        self.task_size = 10
        self.event_queue = []
        self.ue_connections = ue_connections
        self.resources_ue = 5
        self.resources_uav = 10
        self.resources_bs = 150
        self.resources_haps = 100
        self.users = range(ue_connections.shape[0])
        self.processing_times = []
        self.resources_per_servers = []
        self.servers = []
        self.n_bs = n_bs
        self.n_uav = n_uav
        self.n_ues = len(self.users)
        self.uav_offload_prob = 0.5
        self.delay_statistics = {}

    def generate_tasks(self):
        for user_id in self.users:
            arrival_time = 0
            for _ in range(self.num_tasks):
                arrival_time += random.expovariate(self.mean_arrival_rate)
                task = Task(arrival_time, user_id)
                heapq.heappush(self.event_queue, (arrival_time, task))

    def initialize_servers(self):
        for bs in range(self.n_bs):
            self.processing_times.append(self.task_size / self.resources_bs)
            self.resources_per_servers.append(self.resources_bs)
        for uav in range(self.n_uav):
            self.processing_times.append(self.task_size / self.resources_uav)
            self.resources_per_servers.append(self.resources_uav)
        self.processing_times.append(self.task_size / self.resources_haps)
        self.resources_per_servers.append(self.resources_haps)
        num_servers = self.n_bs + self.n_uav + 1
        self.servers = [Server(self.processing_times[i], i + len(self.users),
                               self.resources_per_servers[i]) for i in range(num_servers)]
        self.delay_statistics = {n_s: [] for n_s in range(self.n_ues, self.n_ues + num_servers)}

    def process_task(self, task, strategy_num):
        ue_num = task.user_id
        uav_ids = np.array(range(ue_num+self.n_bs, ue_num+self.n_bs+self.n_uav))
        if strategy_num == 1:
            # best availability strategy
            available_servers = []
            for conn in self.ue_connections[ue_num]:
                k = int(conn - self.n_ues)
                available_servers.append(self.servers[k])

            best_comp_res = available_servers[0].comp_resources
            serving_server = available_servers[0]
            flag_offl_uav = 0
            for s in available_servers:

                if np.any(s.server_id == uav_ids):
                    curr_comp_res = random.choices([self.resources_haps, self.resources_uav],
                                                   [self.uav_offload_prob, 1-self.uav_offload_prob])
                    curr_comp_res = curr_comp_res[0]
                    if curr_comp_res > best_comp_res:
                        serving_server = s
                        best_comp_res = copy.copy(curr_comp_res)

                else:
                    curr_comp_res = s.comp_resources
                    if curr_comp_res > best_comp_res:
                        flag_offl_uav = 0
                        serving_server = s
                        best_comp_res = copy.copy(curr_comp_res)

            if task.arrival_time > serving_server.current_time:
                serving_server.current_time = task.arrival_time
            task.queue_entry_time = copy.copy(task.arrival_time)
            serving_server.tasks_queue.append(task)

            server_id = serving_server.server_id
            prop_delay = self.dist_matrix[ue_num, server_id]/3e8

            # process the first task in the queue
            curr_task = serving_server.tasks_queue[0]
            waiting_time = serving_server.current_time - curr_task.queue_entry_time
            if flag_offl_uav == 0:
                completion_time = serving_server.current_time + serving_server.processing_time + \
                                                 prop_delay + waiting_time
                self.delay_statistics[server_id].append(completion_time - curr_task.arrival_time)
            else:
                prop_delay2 = self.dist_matrix[server_id, self.n_bs + self.n_uav]
                completion_time = serving_server.current_time + self.servers[-1].processing_time \
                                                 + prop_delay + prop_delay2 + waiting_time
                self.delay_statistics[self.n_ues+self.n_bs+self.n_uav].append(completion_time - curr_task.arrival_time)

            print(completion_time - curr_task.arrival_time)
            serving_server.tasks_queue.pop(0)
            serving_server.current_time += self.servers[-1].processing_time

    def run(self):
        self.generate_tasks()
        self.initialize_servers()
        while self.event_queue:
            # event_time, event = self.event_queue[0]
            event_time, event = heapq.heappop(self.event_queue)
            # self.current_time = event_time

            if isinstance(event, Task):
                self.process_task(event, 1)

        #print(self.delay_statistics)
