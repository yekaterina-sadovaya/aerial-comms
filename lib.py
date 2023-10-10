import numpy as np
import math
import random
import copy
import heapq


class AerDeployment:
    def __init__(self):
        self.n_ues = 10
        self.n_uav = 1
        self.R_haps = 1500
        self.n_tasks = int(1e2)
        self.precision = 3
        self.seed = 1
        self.fc_haps = 60e9
        self.fc_bs = 30e9
        self.fc_uav = 40e9
        self.tx_pow_ue = 15
        self.tx_pow_uav = 20
        self.tx_pow_haps = 40
        # nodes are enumerated in the following way 0-n_ue (ues), n_ue-n_ue+n_bs (bss),
        # n_ue+n_bs-n_ue+n_bs+n_uav (uavs), n_ue+n_bs+n_uav-n_ue+n_bs+n_uav+1 (haps)
        self.channel_matrix = np.zeros([self.n_ues + self.n_uav + 1,
                                        self.n_ues + self.n_uav + 1])
        self.propagation_dists = np.zeros([self.n_ues + self.n_uav + 1,
                                           self.n_ues + self.n_uav + 1])
        self.mc_degree = 2
        self.active_connections = np.zeros([self.n_ues, self.mc_degree])
        self.haps_height = 1500
        self.uav_height = 200
        self.ue_height = 1.5
        self.ue_pos = np.zeros([self.n_ues, 3])
        self.haps_pos = np.array([0, 0, self.haps_height])
        self.uav_pos = np.zeros([self.n_uav, 3])
        self.thermal_noise_density = -173.93
        self.conn_info = {'ue'+str(ue): [] for ue in range(self.n_ues)}   # list contains node id and propagation delay

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
        for link1_num in range(self.n_ues + self.n_uav + 1):
            for link2_num in range(self.n_ues + self.n_uav + 1):
                if (link1_num < self.n_ues) and (link2_num < self.n_ues):
                    continue
                else:
                    if link1_num < self.n_ues:
                        tx_pow = self.tx_pow_ue
                        height1 = self.ue_height
                        pos1 = self.ue_pos[link1_num, :]
                    elif self.n_ues <= link1_num < self.n_ues + self.n_uav:
                        tx_pow = self.tx_pow_uav
                        height1 = self.uav_height
                        pos1 = self.ue_pos[self.n_uav - (link1_num - self.n_ues), :]
                    else:
                        tx_pow = self.tx_pow_haps
                        height1 = self.haps_height
                        pos1 = self.haps_pos

                    if link2_num < self.n_ues:
                        height2 = self.ue_height
                        pos2 = self.ue_pos[link2_num, :]
                        fc = self.fc_uav
                    elif self.n_ues <= link2_num < self.n_ues + self.n_uav:
                        height2 = self.uav_height
                        pos2 = self.uav_pos[link2_num - self.n_ues, :]
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
                if (self.n_ues+self.n_uav) > max_id >= self.n_ues:
                    prop_delay = self.propagation_dists[ue, max_id]/299792458
                    self.conn_info['ue'+str(ue)].append(['uav'+str(max_id - self.n_ues), prop_delay])
                elif max_id == (self.n_ues+self.n_uav):
                    prop_delay = self.propagation_dists[ue, max_id]/299792458
                    self.conn_info['ue'+str(ue)].append(['hap', prop_delay])


class Task:
    def __init__(self, arrival_time, user_id):
        self.arrival_time = arrival_time
        self.user_id = user_id
        self.queue_entry_time = 0


class Server:
    def __init__(self, processing_time, server_id, comp_resources, C):
        self.processing_time = processing_time
        self.server_id = server_id
        self.tasks_queue = [[] for _ in range(C)]
        self.comp_resources = comp_resources
        self.current_time = 0


class BasicOffloading:
    def __init__(self, conn_info, num_tasks, mean_arrival_rate, n_uav):
        self.num_tasks = num_tasks
        self.mean_arrival_rate = mean_arrival_rate
        self.r = 10
        self.C = 60
        self.C_UE = 200
        self.C_UAV = 400
        self.C_HAP = 1000
        self.event_queue = []
        self.ue_connections = conn_info
        self.users = range(len(conn_info))
        self.processing_times = {'ues': [], 'uavs':[], 'hap':[]}
        self.servers = []
        self.n_uav = n_uav
        self.n_ues = len(self.users)
        self.delay_statistics = {'ues': [], 'uavs':[], 'hap':[]}
        self.prob_of_local_compute = 0.1
        self.prob_of_offloading_to_UAV = 0.2

    def generate_tasks(self):
        for user_id in self.users:
            arrival_time = 0
            for _ in range(self.num_tasks):
                arrival_time += random.expovariate(self.mean_arrival_rate)
                task = Task(arrival_time, user_id)
                heapq.heappush(self.event_queue, (arrival_time, task))

    def initialize_servers(self):

        for uav in range(self.n_uav):
            t_pr_uav = self.C / self.C_UAV
            self.processing_times['uavs'].append(t_pr_uav)
            self.servers.append(Server(t_pr_uav, 'uav'+str(uav), self.C_UAV, 5))

        for ue in range(self.n_ues):
            t_pr_ue = self.C / self.C_UE
            self.processing_times['ues'].append(t_pr_ue)
            self.servers.append(Server(t_pr_ue, 'ue'+str(ue), self.C_UE, 1))

        t_pr_hap = self.C / self.C_HAP
        self.processing_times['hap'].append(t_pr_hap)
        self.servers.append(Server(t_pr_hap, 'hap', self.C_HAP, 15))

    def process_task(self, task):
        ue_num = task.user_id

        # drop probability of local compute
        no_offl = False
        if random.random() < self.prob_of_local_compute:
            node_name = "ue"
            stat_id = 'ues'
            serving_node = next(item for item in self.servers if item.server_id == node_name+str(ue_num))
            no_offl = True

        # drop probability of offloading to UAV
        offl_to_UAV = False
        if (random.random() < self.prob_of_offloading_to_UAV) and (no_offl == False):
            node_ids = [cn[0] for cn in self.ue_connections['ue'+str(ue_num)] if cn[0][0]=='u']
            node_name = node_ids[0]
            stat_id = 'uavs'
            serving_node = next(item for item in self.servers if item.server_id == node_name)
            offl_to_UAV = True

        if (no_offl == False) and (offl_to_UAV == False):
            node_name = "hap"
            stat_id = 'hap'
            serving_node = next(item for item in self.servers if item.server_id == node_name)

        # append tasks to the shortest queue
        all_q_lens = [len(q) for q in serving_node.tasks_queue]
        all_q_lens = np.array(all_q_lens)
        min_len_id = np.where(all_q_lens == np.min(all_q_lens))
        min_len_id = min_len_id[0][0]
        q_i = serving_node.tasks_queue[min_len_id]
        q_i.append(task)

        # process tasks
        FlagProcessing = 0
        for q in serving_node.tasks_queue:
            if len(q) > 0:
                if serving_node.current_time >= q[0].arrival_time:
                    t_d = serving_node.current_time - q[0].arrival_time
                    self.delay_statistics[stat_id].append(t_d)
                    del q[0]
                    FlagProcessing = 1
                else:
                    serving_node.current_time = q[0].arrival_time

        if (FlagProcessing == 1) or (serving_node.current_time == 0):
            serving_node.current_time += serving_node.processing_time

        return FlagProcessing

    def run(self):
        self.generate_tasks()
        self.initialize_servers()
        while self.event_queue:
            event_time, event = self.event_queue[0]
            self.current_time = event_time

            if isinstance(event, Task):
                wasProcessed = self.process_task(event)
                if wasProcessed == True:
                    heapq.heappop(self.event_queue)

        # print(self.delay_statistics)
        t_star = 1/10
        av_latency = np.array(self.delay_statistics['ues'])
        av_latency = np.append(av_latency, self.delay_statistics['uavs'])
        av_latency = np.append(av_latency, self.delay_statistics['hap'])
        percent_below_threshold = np.sum(av_latency <= t_star)/len(av_latency)
        print('Delay <= t_star: ' + str(percent_below_threshold))
        print('Average Delay: ' + str(np.mean(av_latency)))
