import numpy as np
import matplotlib.pyplot as plt
import pickle

from lib import AerDeployment, BasicOffloading


def compute(ns_UAV, ns_HAP, eps_all, nu_all, file_name):
    # create and configure deployment
    deployment = AerDeployment()
    deployment.drop_nodes()
    deployment.compute_channels()

    perc_stat = []
    x, y = [], []
    for eps_val in eps_all:
        for nu_val in nu_all:
            # create the offloading instance
            offl = BasicOffloading(deployment.conn_info, 10000, 0.5, deployment.n_uav,
                                   eps_val, nu_val, ns_UAV, ns_HAP)
            offl.run()
            perc_stat.append(offl.percent_below_thr)
            x.append(eps_val)
            y.append(nu_val)

    data = {'x': x,
            'y': y,
            'z': perc_stat}

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

    ns_UAV = 5
    ns_HAP = 15
    eps_all = np.linspace(0, 1, 11)
    nu_all = np.linspace(0, 1, 11)
    file_name = 'data.pickle'
    # compute(ns_UAV, ns_HAP, eps_all, nu_all, file_name)
    plot_from_data(file_name)
