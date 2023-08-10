# Geometric Superpixel Representations for Efficient Image Classification with Graph Neural Networks

This repository contains the code for the paper "Geometric Superpixel Representations for Efficient Image Classification with Graph Neural Networks" published at the 4th Visual Inductive Priors for Data-Efficient Deep Learning Workshop at ICCV 2023. It provides a framework for constructing Region Adjacency Graphs (RAGs) based on arbitrary image datasets. The code supports multiple options for the creation of these RAGs and allows training on them efficiently using ShapeGNN. Right now, the project supports the CIFAR-10 and ImageNet-1k datasets. Details can be found in the paper. [[`Paper`](https://openreview.net/pdf?id=E7zgkaEDcE)]

<div>
  <img width="100%" alt="Geometric Superpixel Representations for Efficient Image Classification with Graph Neural Networks illustration" src="images/arch.gif">
</div>

# Table of contents
1. [Abstract](README.md/#Abstract)
1. [Results](#Results)
1. [Structure](#Structure)
1. [Dependencies](#Dependencies)
1. [Setup](#Setup)
1. [Execution](#Execution)
1. [Licensing](#Licensing)

## Abstract
While Convolutional Neural Networks and Vision Transformers are the go-to solutions for image classification, their model sizes make them expensive to train and deploy.
Alternatively, input complexity can be reduced following the intuition that adjacent similar pixels contain redundant information. This prior can be exploited by clustering such pixels into superpixels and connecting adjacent superpixels with edges, resulting in a sparse graph representation on which Graph Neural Networks (GNNs) can operate efficiently. Although previous work clearly highlights the computational efficiency of this approach, this prior can be overly restrictive and, as a result, performance is lacking compared to contemporary dense vision methods. In this work, we propose to extend this prior by incorporating shape information into the individual superpixel representations. This is achieved through a separate, patch-level GNN. Together with enriching the previously explored appearance and pose information of superpixels and further architectural changes, our best model, ShapeGNN, surpasses the previous state-of-the-art in superpixel-based image classification on CIFAR-10 by a significant margin. We also present an optimised pipeline for efficient image-to-graph transformation and show the viability of training end-to-end on high-resolution images on ImageNet-1k.


## Results
### CIFAR-10 results
The results of *ShapeGNN* and its variations on the CIFAR-10 dataset are shown in the table below. The command line arguments corresponding to each of these runs can be found in [ablation_hyperparameters.txt](hparam_files/ablation_hyperparameters.txt). The order of the setups in that file is the same as in the table below. As an example, the arguments used for the "no size" ablation, the fifth entry in the table, are specified in the fifth configuration in [ablation_hyperparameters.txt](hparam_files/ablation_hyperparameters.txt).

| ablation     | accuracy     | median time per epoch |
|--------------|-----------|------------|
| ***ShapeGNN***   | **80.44%**      | **19.4s**        |
| no shapes      | 77.9%  | 12.0s      |
| no shapes, d<sub>g-hidden</sub>=310     | 78.4%  | 12.4s      |
| no colour std. dev.      | 78.1%  | 19.7s      |
| no size      | 80.0%  | 19.7s      |
| SLIC      | 79.3%  | 18.2s      |
| SLIC, no shapes      | 77.6%  | 10.7s      |
| rotation information      | 78.6%  | 19.8s      |
| EGNN      | 58.7%  | 29.5s      |
| DynamicEdgeConv  | 74.8%  | 355.5s     | 
| no position information  | 71.3%  | 18.9s     | 
| no residual connections    | 79.0%  | 18.6s     | 
| concatenation for residual connections   | 79.0%  | 21.2s     | 
| mean pooling    | 79.6%  | 19.5s     | 
| max pooling    | 78.8%  | 19.5s     | 
| sum pooling    | 79.7%  | 19.0s     | 
| d<sub>g-hidden</sub>=150    | 79.9%  | 14.2s     | 
| d<sub>g-hidden</sub>=450, num<sub>g-blocks</sub>=2, num<sub>g-layers</sub>=2    | 80.7%  | 50.7s     | 
| d<sub>l-hidden</sub>=32, num<sub>l-blocks</sub>=1, num<sub>l-layers</sub>=1    | 79.9%  | 16.3s     | 
| d<sub>l-hidden</sub>=128, num<sub>l-blocks</sub>=3, num<sub>l-layers</sub>=3     | 80.2%  | 31.8s     | 
| d<sub>latent</sub>=1    | 79.2%  | 19.6s     | 
| d<sub>latent</sub>=10    | 80.3%  | 19.2s     | 


### ImageNet-1k results
The results of *ShapeGNN* and its variations on the ImageNet-1k dataset are shown in the table below. The command line arguments corresponding to each of these runs can be found in [imagenet_hyperparameters.txt](hparam_files/imagenet_hyperparameters.txt). The order of the setups in that file is the same as in the table below. As an example, the arguments used for the "no shape" ablation, the second entry in the table, are specified in the second configuration in [imagenet_hyperparameters.txt](hparam_files/imagenet_hyperparameters.txt).

| ablation     | accuracy     | median time per epoch | batch size |
|--------------|-----------|------------|------------|
| ***ShapeGNN***   | **46.8%**      | **1045s**        | **480**        |
| no shapes      | 40.1%  | 706s      |    640 |
| num<sub>g-blocks</sub>=2      | 50.4%  | 1430s     |     320 |


## Structure
The structure of this repository is as follows:
- [hparam_files](hparam_files)
  - files with the command line arguments for the different runs
- [runs](runs)
  - pretrained models
- [src](src)
  - source code
- [env.yml](env.yml)
  - environment file for a CPU-based Anaconda environment
- [env_gpu.yml](env_gpu.yml)
  - environment file for a GPU-based Anaconda environment
  

## Dependencies
The code was tested to run on Linux. For other operating systems, the [CMakeLists.txt](src/cpp_preprocessing/CMakeLists.txt) would most likely need
to be modified to link to the appropriate threading and math libraries.
The code uses the OpenCV framework (both opencv and opencv-contrib need to be installed in such a way that C++ headers and corresponding implementations are available (see e.g. https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html).

Other dependencies necessary are a recent [GCC](https://gcc.gnu.org/) compiler (GCC 10.3 was used), [CMake](https://cmake.org/) (CMake 3.20 was used), and [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) (Anaconda 2021.05 was used).

After installing these dependencies, you need to install one of the two environments [env.yml](env.yml) or [env_gpu.yml](env_gpu.yml) (depending on if you have a GPU or not).

You can do so with a command like the following:

```conda env create -f env_gpu.yml```

## Setup

Activate the previously installed environment with:
```conda activate gnn_img_cls_gpu```

Assuming you are in the environment and all other dependencies are installed, you should be able to compile the C++ image to graph transformation:

```cd src/cpp_preprocessing```

```mkdir build```

```cd build```

```cmake -DCMAKE_BUILD_TYPE=Release -DINCLUDE_SLIC=True .. && cmake --build . --verbose && cd ..```

If the dependencies were installed and you are in the environment but the compilation errors out, please check the [CMakeLists.txt](src/cpp_preprocessing/CMakeLists.txt) and remove the options marked as for optimisation (although this should not happen).

Once an executable file is created, you can run the main Python code (which will call the executable file itself). 

For further information about the hyperparameters, use:
```python3 main.py --help```

The hyperparameter configurations used in the paper are specified in the [hparam_files](hparam_files) folder.

## Execution

To reduce runtime, existing preprocessing or model results are loaded if they are available:
- Please note that the Python code will not construct the dataset again if it already exists (for that specific preprocessing parameter configuration).
If you want it to reconstruct it, please remove the dataset folder.
- Please also note that if you run a model two times, and at least one epoch has finished the first time the model was trained, the model will be loaded and run on the test set. To prevent this, remove the first model folder in ```runs``` before running it a second time.
- If the model had finished training and a result on the test set was reported, that result will be loaded the next time the model is ran. To prevent this and recompute the testing accuracy, remove the test_results.json file from the corresponding model's folder.

We provide pretrained models for ShapeGNN for both CIFAR-10 and ImageNet-1k, so running the ShapeGNN with the parameters mentioned
in the first parameter line of the corresponding hyperparameter files (the one with the "# ShapeGNN" comment above it) will just load the test results. **If you want to reproduce the testing accuracies for the provided pretrained models, please remove the test_results.json file in the corresponding model's folder and then run the model**.

In the paper, we report the median time per epoch during training. This information is printed after completing the training of a new model. For an already trained model, it can be manually computed. The duration of each epoch is saved in the fourth row in the model_results.csv in a model's directory.

### CIFAR-10

For CIFAR-10, you should just be able to run the code directly without downloading the dataset beforehand (the download is done automatically):

```python3 main.py --dataset=cifar10 --num_workers=0 --disable_val --norm_centroids --normalise_shapes --use_stddev --graph_of_graphs```

### ImageNet-1k

For ImageNet-1k, the dataset needs to be downloaded and have the following structure:

- at the root level of the dataset, there needs to be a ```train``` and a ```val``` folder. Each of these folders contains one folder per class, with the `id` of the class as the folder name (e.g. "n02132136"). Each class folder contains the images of that class, whose names are of the form `id_nr.JPEG`, e.g. `n02132136_30976.JPEG`.
- at the root level of the dataset, there also needs to be a `ILSVRC2012_devkit_t12` folder, which has a `data` folder in it, which has a `meta.mat` file in it. This file should come with the ImageNet-1k download and is from where the class name to label mapping is taken from. 

For the preprocessing, you can also use a separate run of the Python code that finishes right after the dataset was preprocessed. This is useful if you want to preprocess the dataset with the maximum amount of RAM available by passing `max_out_ram_for_preprocessing` as a command line option:

```python3 main.py --dataset=imagenet --num_workers=0 --scale_params_to_img --max_out_ram_for_preprocessing --only_create_dataset --scale_only_min_size --scale 300 --min_size_patch 0.12 --approx_epsilon 0.07```

This option was tuned for a RAM size of 96GB and for the presented ImageNet-1k preprocessing parameters.

For ImageNet-1k, if the dataset is already in this format somewhere else on the disk (such as in "/datasets/imagenet"), you can pass the path to the root `imagenet` folder as a command line parameter to the Python program:

```python3 main.py --orig_data_dir /datasets/imagenet --dataset=imagenet --num_workers=2  --epochs=100 --batch_size=480 --disable_val --norm_centroids --normalise_shapes --use_stddev --scale_params_to_img --scale_only_min_size --scale 300 --min_size 0.12 --approx_epsilon 0.07 --graph_of_graphs```

# Citation
If you find this repository useful, please consider citing our paper:
```
@inproceedings{cosma2023geometric,
  title={Geometric Superpixel Representations for Efficient Image Classification with Graph Neural Networks},
  author={Cosma, Radu Alexandru and Knobel, Lukas and Van der Linden, Putri A and Knigge, David M and Bekkers, Erik J},
  booktitle={4th Visual Inductive Priors for Data-Efficient Deep Learning Workshop},
  year={2023}
}
```

## Licensing
The code is licensed as MIT with the exception of files that were adapted from other sources (all permissive open source licenses), whose licenses can be found at the top of the respective files.
