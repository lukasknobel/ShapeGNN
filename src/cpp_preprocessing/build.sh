#!/bin/bash
cd build
module purge
module load 2021
module load Anaconda3/2021.05
module load GCC/10.3.0
module load CMake/3.20.1-GCCcore-10.3.0
module load OpenCV/4.5.3-foss-2021a-CUDA-11.3.1-contrib

source activate dl2

cmake -DCMAKE_BUILD_TYPE=Release -DINCLUDE_SLIC=True .. && cmake --build . --verbose
cd ..
