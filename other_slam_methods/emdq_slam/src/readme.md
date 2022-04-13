# EMDQ Source Code

## Original Code
https://github.com/haoyinzhou/EMDQ_C

C++ code for the EMDQ algorithm, which can remove mismatches from SURF/ORB feature matches and also generate smooth and dense deformation field.

Paper:  Haoyin Zhou, Jagadeesan Jayender, "EMDQ: Removal of Image Feature Mismatches in Real-Time", IEEE Transactions on Image Processing, 2021

usage: Usage: ./demoEMDQ PathToImage1 PathToImage2
For example: ./demoEMDQ picture1.png picture2.png

Dependencies: Eigen, OpenCV, and OpenMP (optional)

## Changes
We added a python wrapper based on boost-python.



