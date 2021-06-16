# Detection and Segmentation of coryphaena in auction boxes
A reduced version of the scripts used in the work (ref article), is available to train, test and use over different dataset. 
All the dataset is stored in folder dataset.

THIS BRANCH HAS BEEN DEVELOPED BY THE USER [ESTEVE VALLS MASCARO](https://github.com/Evm7) 💻


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
python demo_all.py -mode <MODE> [-directory <DIRECTORY>] [--weigths_path <weigth_path_file.h5>] [--not_display]  [--save] [--notconfussion] [--epochs EPOCHS] [--image_num IMAGE_NUM] [--layers LAYERS] ...
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
- '--train_annotations'  --> default="/dataset/train2.json", help='Introduce the path to the training dataset'
- '--train_images'  --> default="/dataset/img_train", help='Introduce the path to the training images'
- '--val_annotations' --> default="'/dataset/val.json'", help='Introduce the path to the validation dataset'
- '--val_images' --> default="'/dataset/val.json'", help='Introduce the path to the validation dataset'¡


## Scheme:
The project has been divided into a deterministic scheme:
  - mrcnn/: obtained from the MaskRcnn Matterplot repository, it contains the definition of the neural netwowrk which will be used in this project.
  - dataset/: it contains the annotated files with the tagged images references, and the images. It contains all the information for the training, evaluating and testing.
  - weigths/: created when training the network (.h5).
  - dataset.py : contains the classes used to manage the annotation files. It provides functions to both load dataset and loads configurations. Subclass from the Mask Rcnn matterplot. 
  - demo_all.py : main file which contains the functions to train, test and evaluate the neural network.
  - inference/: contains the processed images with the masks detected in the inference.

