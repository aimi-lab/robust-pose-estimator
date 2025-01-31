<!-- title -->
# Robust Camera Pose Estimation for Endoscopic Videos

Visual Odometry for stereo endoscopic videos with breathing and tool deformations. For more details, please see our paper:
[Learning how to robustly estimate camera pose in endoscopic videos](https://arxiv.org/abs/2304.08023v1) 

Please cite us if you use our code for your research:

```
@article{hayoz2023pose,
title={Learning how to robustly estimate camera pose in endoscopic videos},
author={Michel Hayoz, Christopher Hahne, Mathias Gallardo, Daniel Candinas, Thomas Kurmann, Max Allan, Raphael Sznitman},
journal={International Journal of Computer Assisted Radiology and Surgery 2023}
doi={https://doi.org/10.1007/s11548-023-02919-w}
}
```

![Alt text](./system_overview.png)
## Installation

you will need Python 3.8, which can be checked with

``` $ python3 --version ```

install all requirements with

``` $ pip install -r requirements.txt ```

for optional 3D visualizations install open3D with

``` $ pip install open3d ```

checkout thirdparty code with

``` $ git submodule update --init --recursive ```

``` $ pip install git+https://github.com/princeton-vl/lietorch.git ```

## Prepare the data
1. Download the *StereoMIS* dataset from [here](https://doi.org/10.5281/zenodo.7727691) and unzip it in the data folder.
Note, only provide the porcine sequences of *StereMIS* are public. Each sequence contains a stereo video file, 
camera calibration and camera poses.
2. For training and (fast) inference you need to unpack and pre-process the stereo video files with:
``` 
$ cd scripts
$ python preprocess_video_data.py ../data/StereoMIS
```    

## Training
Train the pose estimator with
``` 
$ cd scripts
$ python train_posenet.py
```
you may need to adapt the training parameters in the configuration file in *configuration/train.yaml*   

## Inference
You can infer the camera trajectory for a sequence with were the data needs to be in the same format as the downloaded StereoMIS
``` 
$ cd scripts
$ python infer_trajectory.py path/to/sequence
```

benchmark on surgical scenarios with (only works on pre-processed StereoMIS data)
``` 
$ cd scripts
$ python benchmark_scenarios.py path/to/test_set
```

benchmark on test set with (only works on pre-processed StereoMIS data)
``` 
$ cd scripts
$ python benchmark_test.py path/to/test_set
```


