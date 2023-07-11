from lib import AerDeployment

if __name__ == '__main__':
    deployment = AerDeployment()
    deployment.generate_times()
    deployment.drop_nodes()
    deployment.compute_channels()
    print()