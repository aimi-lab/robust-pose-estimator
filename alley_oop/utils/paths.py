from pathlib import Path

SCARED_ROOT_PATH = Path('/home/chris/UbelixWorkspaces/artorg_aimi/ws_00000/innosuisse_surgical_robot/01_Datasets/01_stereo/01_intuitive_scared/')


def get_scared_relpath(d_idx:int=1, k_idx:int=1, fname:str=''):

    # account for test dataset names
    if d_idx < 8:
        data_path = Path('dataset_' + str(d_idx))
    else:
        try:
            data_path = Path('dataset_' + str(d_idx))
        except FileNotFoundError:
            data_path = Path('test_dataset_' + str(d_idx))
        k_idx -= 1

    return data_path / Path('keyframe_' + str(k_idx)) / fname


def get_scared_abspath(d_idx:int=1, k_idx:int=1, fname:str=''):
    return Path(SCARED_ROOT_PATH) / get_scared_relpath(d_idx, k_idx, fname)


def get_save_path(d_idx:int=1, k_idx:int=1, base_path:str=None):

    base_path = Path().cwd() / 'generated_data' if base_path is None else Path(base_path)
    save_path = base_path / get_scared_relpath(d_idx, k_idx)
    save_path.mkdir(parents=True, exist_ok=True)

    return save_path


def str2path(path:str=None):
    return Path(path)
