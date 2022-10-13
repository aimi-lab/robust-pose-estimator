import os
from other_slam_methods.elfusion_get_trajectory import main as elfusion
from other_slam_methods.orbslam2_get_trajectory import main as orbslam
import yaml
from multiprocessing import Process
import multiprocessing as mp
import shutil

PATH_REPLACEMENT = ['/home/mhayoz/research', '/storage/workspaces/artorg_aimi/ws_00000']
if __name__ == '__main__':
    with open(os.path.join('..', 'intuitive_slam_test.txt'), 'r') as f:
        sequences = f.readlines()
    mp.set_start_method('spawn')
    for s in sequences:
        sequence, step = s.split(' ')
        step = int(step)
        print(sequence)
        sequence = sequence.replace(PATH_REPLACEMENT[1], PATH_REPLACEMENT[0])
        sequence = sequence.replace('\n', '')
        try:
            # if not os.path.isfile(os.path.join(sequence, 'data/efusion/trajectory.json')):
            #     with open('configuration/efusion_scared.yaml', 'r') as ymlfile:
            #         config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
            #     p = Process(target=elfusion, args=(sequence, os.path.join(sequence, 'data/efusion'), config, 'cpu', 0, 10000000, 1, 'efusion', False))
            #     p.start()
            #     p.join()
            if not os.path.isfile(os.path.join(sequence, 'data/orbslam2/trajectory.json')):
                with open('configuration/orbslam2.yaml', 'r') as ymlfile:
                    config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
                p = Process(target=orbslam, args=(sequence, os.path.join(sequence, 'data/orbslam2'), config, 'cpu', 0, 1000000000, step, 'test_orbslam2', False,))
                p.start()
                p.join()
        except:
            print('not successfull. -> skip')
