import numpy as np
import math
import random
import copy


class AerDeployment:
    def __init__(self):
        self.n_ues = 10
        self.n_bs = 2
        self.n_uav = 2
        self.R_haps = 500
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
        self.tasks_state = {ue: np.zeros([4, self.n_tasks]) for ue in range(10)}    # columns: time, node_num, status, delay
        # nodes are enumerated in the following way 0-n_ue (ues), n_ue-n_ue+n_bs (bss),
        # n_ue+n_bs-n_ue+n_bs+n_uav (uavs), n_ue+n_bs+n_uav-n_ue+n_bs+n_uav+1 (haps)
        self.channel_matrix = np.zeros([self.n_ues + self.n_bs + self.n_uav + 1,
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

    def next_time(self):
        random.seed(self.seed)
        return -math.log(1.0 - random.random()) / self.rate_parameter

    def generate_times(self):
        for ue in range(self.n_ues):
            for i in range(self.n_tasks):
                self.tasks_state[ue][0, i] = np.round(self.next_time(), self.precision)
            self.tasks_state[ue] = np.sort(self.tasks_state[ue])

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
                        pos2 = self.ue_pos[self.n_bs - (link2_num - self.n_ues), :]
                        fc = self.fc_bs
                    elif self.n_ues + self.n_bs <= link2_num < self.n_ues + self.n_bs + self.n_uav:
                        height2 = self.uav_height
                        pos2 = self.ue_pos[self.n_uav - (link2_num - self.n_ues - self.n_bs), :]
                        fc = self.fc_uav
                    else:
                        height2 = self.haps_height
                        pos2 = self.haps_pos
                        fc = self.fc_haps

                    dist = np.linalg.norm(pos1 - pos2)
                    rx_pow = tx_pow - self.UMa_LoS(dist, fc, height1, height2) - self.thermal_noise_density
                    self.channel_matrix[link1_num, link2_num] = 10.0 ** ((rx_pow-30) / 10.0)

        for ue in range(self.n_ues):
            ch = copy.copy(self.channel_matrix[ue, :])
            for n in range(self.mc_degree):
                max_num = max(ch)
                max_id = np.where(ch == max_num)[0][0]
                ch[max_id] = 0
                self.active_connections[ue, n] = max_id