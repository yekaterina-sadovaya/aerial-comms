from lib import AerDeployment, BasicOffloading

if __name__ == '__main__':
    # create and configure deployment
    deployment = AerDeployment()
    deployment.drop_nodes()
    deployment.compute_channels()

    # create the offloading instance
    offl = BasicOffloading(deployment.active_connections, deployment.propagation_dists,
                           1000, 1.5, deployment.n_bs, deployment.n_uav)
    offl.run()
