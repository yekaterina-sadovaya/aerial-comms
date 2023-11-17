import numpy as np
import matplotlib.pyplot as plt
import pickle

from lib import AerDeployment, BasicOffloading


def compute(n_ues, ns_UAV, ns_HAP, eps_all, nu_all, file_name, strategy_number):
    # create and configure deployment
    deployment = AerDeployment(n_ues)
    deployment.drop_nodes()
    deployment.compute_channels()

    perc_stat = []
    x, y = [], []
    for eps_val in eps_all:
        if strategy_number == 1:
            for nu_val in nu_all:
                # create the offloading instance
                offl = BasicOffloading(deployment.conn_info, 1000, 2, deployment.n_uav,
                                       eps_val, nu_val, ns_UAV, ns_HAP, strategy_number)
                offl.run()
                perc_stat.append(offl.percent_below_thr)
                x.append(eps_val)
                y.append(nu_val)
        elif strategy_number == 4:
            offl = BasicOffloading(deployment.conn_info, 1000, 2, deployment.n_uav,
                                   1.0, 0.0, ns_UAV, ns_HAP, 1)
            offl.run()
            perc_stat.append(offl.percent_below_thr)
            x.append(1.0)
            y.append(0.0)
        else:
            offl = BasicOffloading(deployment.conn_info, 1000, 2, deployment.n_uav,
                                   eps_val, 0.1, ns_UAV, ns_HAP, strategy_number)
            offl.run()
            perc_stat.append(offl.percent_below_thr)
            x.append(eps_val)
            for nu_val in nu_all:
                y.append(nu_val)

        data = {'x': x,
                'y': y,
                'z': perc_stat,
                'delays': offl.delay_statistics}

        with open(file_name, 'wb') as file:
            # Serialize and write the variable to the file
            pickle.dump(data, file)


def plot_from_data(file_name):
    with open(file_name, 'rb') as file:
        loaded_data = pickle.load(file)
    x = loaded_data['x']
    y = loaded_data['y']
    z = loaded_data['z']
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    #z = np.vstack([z,z])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('$Nu$')
    ax.set_zlabel('$Perc. < Thr$')
    plt.show()


if __name__ == '__main__':

    #n_ues = 30
    ns_UAV = 5
    ns_HAP = 15
    strategy_number = [2, 3]
    # eps_all = np.linspace(0, 1, 11)
    # nu_all = np.linspace(0, 1, 11)
    eps_all = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    #eps_all = [1.0]
    nu_all = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    n_all = [60, 90, 120, 200]
    for st in strategy_number:
        for n_ues in n_all:
            file_name = 'data_'+str(n_ues)+'_str'+str(st)+'_r2.pickle'
            compute(n_ues, ns_UAV, ns_HAP, eps_all, nu_all, file_name, st)
    #plot_from_data(file_name)
