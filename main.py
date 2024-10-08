import numpy as np
import matplotlib.pyplot as plt
import pickle

from lib import AerDeployment, BasicOffloading, ecdf


def compute(n_ues, ns_UAV, ns_HAP, eps_val, nu_val, file_name, strategy_number):
    # create and configure deployment
    deployment = AerDeployment(n_ues)
    deployment.drop_nodes()
    deployment.compute_channels()

    perc_stat = []
    x, y = [], []

    if strategy_number == 1:

        # create the offloading instance
        offl = BasicOffloading(deployment.conn_info, 1000, 3, deployment.n_uav,
                               eps_val, nu_val, ns_UAV, ns_HAP, strategy_number)
        offl.run()
        perc_stat.append(offl.percent_below_thr)
        x.append(eps_val)
        y.append(nu_val)

    elif strategy_number == 4:
        offl = BasicOffloading(deployment.conn_info, 1000, 3, deployment.n_uav,
                               1.0, 0.0, ns_UAV, ns_HAP, 1)
        offl.run()
        perc_stat.append(offl.percent_below_thr)
        x.append(1.0)
        y.append(0.0)
    else:
        offl = BasicOffloading(deployment.conn_info, 1000, 3, deployment.n_uav,
                               eps_val, 0.1, ns_UAV, ns_HAP, strategy_number)
        offl.run()
        perc_stat.append(offl.percent_below_thr)
        x.append(eps_val)
        y.append(nu_val)

    data = {'eps': x,
            'nu': y,
            'perc': perc_stat,
            'delays': offl.delay_statistics}

    with open(file_name, 'wb') as file:
        # Serialize and write the variable to the file
        pickle.dump(data, file)


def plot_from_data(file_name):
    with open(file_name, 'rb') as file:
        loaded_data = pickle.load(file)

    ecdf(loaded_data['delays']['ues'])
    ecdf(loaded_data['delays']['uavs'])
    ecdf(loaded_data['delays']['hap'])
    plt.legend(['UEs', 'UAVs', 'HAP'])
    plt.xlabel('Delay, s')
    plt.ylabel('CDF')
    plt.grid()
    plt.xlim([0, 1.5])
    plt.ylim([0, 1])
    plt.show()


if __name__ == '__main__':

    # Set number of UEs and UAVs in the deployment
    number_of_ues = 30
    number_of_uav = 3

    # Set number of tasks generated by UEs and UAVs
    number_of_tasks_UAV = 5
    number_of_tasks_HAP = 15

    # Set the strategy, where
    # 1 stands for the baseline strategy
    # 2 stands for the replication strategy
    # 3 stands for the splitting strategy
    # 4 stands for the strategy with no offloading
    strategy_number = 1

    # Set probability of local compute (eps)
    eps_all = 0.4

    # Set probability of offloading to UAV; the probability of offloading to HAPS is then computed as 1-(eps+nu)
    nu_all = 0.1

    # Set filename
    file_name = 'data_str_num_' + str(strategy_number) + '.pickle'

    compute(number_of_ues, number_of_uav, number_of_tasks_HAP, eps_all, nu_all, file_name, strategy_number)

    # Plot results
    plot_from_data(file_name)
