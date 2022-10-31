import os
from other_slam_methods.elfusion_get_trajectory import main as elfusion
from other_slam_methods.orbslam2_get_trajectory import main as orbslam
import yaml
from multiprocessing import Process
import multiprocessing as mp
import shutil
import pandas as pd


PATH_REPLACEMENT = ['/home/mhayoz/research', '/storage/workspaces/artorg_aimi/ws_00000']
if __name__ == '__main__':
    with open(os.path.join('..', 'intuitive_slam_scenarios.txt'), 'r') as f:
        sequences = f.readlines()
    mp.set_start_method('spawn')
    for s in sequences:
        sequence, step = s.split(' ')
        step = int(step)
        print(sequence)
        sequence = sequence.replace(PATH_REPLACEMENT[1], PATH_REPLACEMENT[0])
        sequence = sequence.replace('\n', '')
        assert os.path.isfile(os.path.join(sequence, 'scenarios.csv'))
        df = pd.read_csv(os.path.join(sequence, 'scenarios.csv'))
        for i, row in df.iterrows():
            start = row['start']
            stop = min(row['start'] + 300, row['end'])
            print(f'{start} -> {stop} : {row["scenario"]}')
            try:
                if not os.path.isfile(os.path.join(sequence, f'data/{i}/efusion/trajectory.freiburg')):
                    with open('configuration/efusion_slam.yaml', 'r') as ymlfile:
                        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
                    config.update({'scenario': row['scenario'], 'start': start, 'seq_number': i})
                    p = Process(target=elfusion, args=(sequence, os.path.join(sequence, f'data/{i}/efusion'), config, 'cpu', start, stop, step, 'scenario_efusion', False,))
                    p.start()
                    p.join()
                if not os.path.isfile(os.path.join(sequence, f'data/{i}/orbslam2/trajectory.freiburg')):
                    with open('configuration/orbslam2.yaml', 'r') as ymlfile:
                        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
                    config.update({'scenario': row['scenario'], 'start': start, 'seq_number': i})
                    p = Process(target=orbslam, args=(sequence, os.path.join(sequence, f'data/{i}/orbslam2'), config, 'cpu', start, stop, step, 'scenario_orbslam2', False,))
                    p.start()
                    p.join()
            except:
                print('not successfull. -> skip')
