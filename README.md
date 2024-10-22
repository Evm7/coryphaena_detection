# CoryPhaena Detection
Project CoryPhaena Detection as a supervisor in IMEDEA: training, evaluating and testing of a neural network 

THIS BRANCH HAS BEEN DEVELOPED BY THE USER [ESTEVE VALLS MASCARO](https://github.com/Evm7) 💻

## Work stil in progress ⏳ :

- [X] Add storage capacity and retrieval of the files
- [X] Inference in the metadata of the images from the each day auctions
- [X] Improve Automatiscm
- [X] Creation of automatic structurized system : each image tested is saved in particular directory (independence of day-month-year)
- [X] Optimize Neural Netowrk managing functions
- [X] Creation of elegant and simple way to output the matrix of confussion and some evaluating techniques
- [x] Create inference script ready to be executed automatically daily.
- [X] Improve performance of the CNN by modifying training parameters



## Goal:

The use of underwater cameras for scientific purposes is spreading. IMEDEA (Mediterranean Institute for Advanced Studies) has an extensive collection of underwater images taken over the last 10 years, as well as a three-camera underwater fixed station that collects images and video on a daily basis. All this information is potentially very valuable both to understand ecological processes and to improve the management of the natural marine resources of the Balearic Islands, but extracting it involves a great economic and time cost that ends up being subject to manual processing by specialized personnel.
DEEP LEARNING applications (multi-layered convolution neural networks) have supposed a qualitative advance in per-computational vision. Currently, applications are beyond human capacity, both in efficiency and reliability. So much so that it is widely used in fields such as medicine and industry.
But the applications in marine environment are still scarce. This project proposes (1) to develop a reliable way for the unsupervised interpretation of underwater images and (2) to demonstrate its scientific potential in a case of study  filled in the context of the long-term environmental monitoring strategy. 
The project its being promoted by the Government of the Balearic Islands.

## Pre-requisites
1) Python 3.6
2) Python3-Dev (For Ubuntu, `sudo apt-get install python3-dev`)
3) Numpy `pip3 install numpy`
4) Cython `pip3 install cython`
5) TensorFlow > 1.3.0 --> 1.12.0
6) pip install scikit-learn
7) pip install seaborn
8) Keras == 2.2.5
9) PyCocoTools
10) CUDA/9.0 availability and CUDNN/7.4

```
NOTE: Make sure CUDA_HOME environment variable is set.
```

## Usage:

There is plenty of arguments variations that can modificate the functioning of the video detector:
```
python demo_llampuga.py -mode <MODE> [-directory <DIRECTORY>] [--weigths_path <weigth_path_file.h5>] [--not_display]  [--save] [--notconfussion] [--epochs EPOCHS] [--image_num IMAGE_NUM] [--layers LAYERS] ...
```

- "--directory" -->"Path to the directory of images to be tested"
- '--not_display' -->'Introduce this argument to avoid displaying the image
- '--save' -->'Introduce this argument to save the processed images
- '--notconfussion' -->'Whether to show the mattrix of confussion or not [Default:yes]'
- '--mode' -->'Introduce the aim of the execution: training, evaluating or testing'
- '--weigths_path' -->'Introduce the path to the weigths'
- '--epochs' -->'Introduce the number of epochs for the training'
- '--image_num' -->'Introduce the number of images you want to process'
- '--layers' -->'Introduce the layers tou want to train: either "all" or "heads"'
- '--train_dataset'  --> default="/dataset/train3_merged_190_prueba.json", help='Introduce the path to the training dataset'
- '--val_dataset' --> default="'/dataset(annotations/val2_merged.json'", help='Introduce the path to the validation dataset'
- '--test_dataset_images' --> default='/datset/test_llampuga_todas_cerca_fechas_training/', help='Introduce the path to the testing dataste image directory'
- '--test_dataset_file'  --> default='/dataset/coco_test.json', help='Introduce the path to the testing dataset json file'

## Scheme:
The project has been divided into a deterministic scheme:
  - mrcnn/: obtained from the MaskRcnn Matterplot repository, it contains the definition of the neural netwowrk which will be used in this project.
  - dataset/: it contains the annotated files with the tagged images references. It contains all the information for the training, evaluating and testing.
  - weigths/: created when training the network (.h5).
  - images/: inside there are the directories of the fish auction with names that follow the next rule: /OPMM_Subasta_YYYY_MM_DD/*
  - dataset.py : contains the classes used to manage the annotation files. It provides functions to both load dataset and loads configurations. Subclass from the Mask Rcnn matterplot. 
  - demo_llampuga.py : main file which contains the functions to train, test and evaluate the neural network.
  - inference/: contains the processed images with the masks detected in the inference. It does organize the images by year-month-day

