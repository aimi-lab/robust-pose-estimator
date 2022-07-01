import os
from other_slam_methods.elfusion_get_trajectory import main as elfusion
from other_slam_methods.orbslam2_get_trajectory import main as orbslam
import yaml
from multiprocessing import Process
import multiprocessing as mp

PATH_REPLACEMENT = ['/home/mhayoz/research', '/storage/workspaces/artorg_aimi/ws_00000']
if __name__ == '__main__':
    with open(os.path.join('..', 'scared.txt'), 'r') as f:
        sequences = f.readlines()
    mp.set_start_method('spawn')
    for sequence in sequences:
        print(sequence)

        try:
            sequence = sequence.replace(PATH_REPLACEMENT[1], PATH_REPLACEMENT[0])
            sequence = sequence.replace('\n', '')
            with open('configuration/efusion_scared.yaml', 'r') as ymlfile:
                config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
            # if os.path.isfile(os.path.join(sequence, 'data/efusion/trajectory.json')):
            #     continue
            #elfusion(sequence, os.path.join(sequence, '/data/efusion'), config, 'cpu', 0, 10000000000, 1, None)#, 'efusion')

            with open('configuration/orbslam2.yaml', 'r') as ymlfile:
                config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
            p = Process(target=orbslam, args=(sequence, os.path.join(sequence, 'data/efusion'), config, 'cpu', 0, 1000000000, 1, 'orbslam2', False,))
            p.start()
            p.join()
        except:
            print('not successfull. -> skip')
