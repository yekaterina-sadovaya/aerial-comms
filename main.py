from lib import AerDeployment, BasicOffloading

if __name__ == '__main__':
    # create and configure deployment
    deployment = AerDeployment()
    deployment.drop_nodes()
    deployment.compute_channels()

    # create the offloading instance
    offl = BasicOffloading(deployment.conn_info, 1000, 0.5, deployment.n_uav)
    offl.run()
