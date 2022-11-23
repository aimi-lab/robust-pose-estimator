from pathlib import Path

from core.utils.const_custom import MY_WORKSPACE_PATH
from core.utils.const_global import SCARED_UBLX_PATH, SEGMEN_UBLX_PATH

SCARED_ROOT_PATH = Path(MY_WORKSPACE_PATH) / SCARED_UBLX_PATH
SEGMEN_ROOT_PATH = Path(MY_WORKSPACE_PATH) / SEGMEN_UBLX_PATH


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
