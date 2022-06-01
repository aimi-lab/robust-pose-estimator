<!-- title -->
# ALLEY - <img src="./tests/test_data/bball.jpeg" alt="o" width="28"/><img src="./tests/test_data/bball.jpeg" alt="o" width="28"/>P

*alley-oop* is a collection of SLAM routines.

[![tests @ develop](https://github.com/aimi-lab/alley-oop/workflows/tests/badge.svg?branch=develop&event=push)](https://github.com/aimi-lab/alley-oop/actions/workflows/tests.yaml)

## Installation

you will need Python 3.8, which can be checked with

``` $ python3 --version ```

Installation can be accomplished using the following bash script

``` $ bash install_env.sh```

**Note**

if the above install script is not used, submodules can be loaded manually by

``` $ git submodule update --init --recursive ```

## Conventions

- image ```torch.Tensor``` dimensions: N x C x H x W
- point array/tensor dimensions: 3 x M

## Run Alley-OOP SLAM

    cd scripts
    python alleyoop_get_trajectory.py path/to/input/folder
