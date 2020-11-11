import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

import imgaug as ia
from imgaug import augmenters as iaa
import keras
import json

import mrcnn.utils as utils
import mrcnn.model as modellib
import mrcnn.visualize as visualize
from mrcnn.model import log as log
import inspect
import mrcnn.config as config

from datetime import *

from numpy.distutils.system_info import numarray_info

from dataset_toni import DatasetConfig

import pandas as pd


class Demo_toni():

    def initialization(self, weight_path="weights/mask_rcnn_coco.h5"):
        # Root directory of the project
        self.ROOT_DIR = os.path.abspath("")
        print("ROOT DIR:" + self.ROOT_DIR)
        # Import Mask RCNN
        sys.path.append("")  # To find local version of the library
        os.chdir("mrcnn")
        print("mrcnn:" + os.getcwd())
        module = inspect.getmodule(config)
        module_path = os.path.dirname(module.__file__)
        print("MODULE PATH:" + module_path)

        # Directory to save logs and trained model
        self.MODEL_DIR = os.path.join(self.ROOT_DIR, "logs")
        self.create_directory(self.MODEL_DIR)
        print("MODEL_DIR PATH:" + self.MODEL_DIR)

        self.DATASET_DIR = os.path.join(self.ROOT_DIR, "dataset")
        self.create_directory(self.DATASET_DIR)
        print("DATASET_DIR PATH:" + self.DATASET_DIR)

        self.WEIGTHS_DIR = os.path.join(self.ROOT_DIR, "weights")
        self.create_directory(self.WEIGTHS_DIR)
        print("WEIGTHS_DIR PATH:" + self.WEIGTHS_DIR)

        self.IMAGES_DIR = os.path.join(self.ROOT_DIR, "images")
        self.create_directory(self.IMAGES_DIR)
        print("IMAGES_DIR PATH:" + self.IMAGES_DIR)

        self.INFERENCE_DIR = os.path.join(self.ROOT_DIR, "inference")
        self.create_directory(self.INFERENCE_DIR)
        print("INFERENCE_DIR PATH:" + self.INFERENCE_DIR)

        self.LOGS_DIR = os.path.join(self.ROOT_DIR, "logs")
        self.create_directory(self.LOGS_DIR)
        print("LOGS_DIR PATH:" + self.LOGS_DIR)

        # Local path to trained weights file
        self.COCO_MODEL_PATH = os.path.join(self.ROOT_DIR, "weights/mask_rcnn_coco.h5")
        print("COCO_MODEL_PATH:" + self.COCO_MODEL_PATH)

        if weight_path == None:
            # Download COCO trained weights from Releases if needed
            utils.download_trained_weights(self.COCO_MODEL_PATH)

    def configuration(self):
        self.config = DatasetConfig()
        self.config.display()

        self.date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        print("Date" + str(self.date))
        try:
            self.config.save(self.MODEL_DIR, str(self.date))
        except FileNotFoundError:
            print("File /logs/" + str(self.date) + " does not exist. We proceed to create so")
            f = open(self.MODEL_DIR + "/Config_" + str(self.date) + ".txt", "x+")
            f.write("Configurations")
            f.close()
            self.config.save(self.MODEL_DIR, str(self.date))

    def initialize_dataset(self, mode, training_dataset, validation_dataset, testing_dataset, visualize_images=False,
                           llotja=False):
        from dataset import Dataset
        if llotja:
            self.dataset_val = Dataset()
            self.dataset_val.load_dataset('', self.DATASET_DIR + "/" + validation_dataset)
            self.dataset_val.prepare()
            return
        if mode == 'training':
            # Training dataset
            self.dataset_train = Dataset()
            self.dataset_train.load_dataset('', self.DATASET_DIR + "/" + training_dataset)
            self.dataset_train.prepare()
            # Validation dataset
            self.dataset_val = Dataset()
            self.dataset_val.load_dataset('', self.DATASET_DIR + "/" + validation_dataset)
            self.dataset_val.prepare()

            # Check Training dataset Annotations
            if visualize_images:
                dataset = self.dataset_train
                image_ids = np.random.choice(dataset.image_ids, 3)  # np.array(dataset.image_ids)
                print(image_ids)

                for image_id in image_ids:
                    image = dataset.load_image(image_id)
                    mask, class_ids = dataset.load_mask(image_id)
                    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
                    print(class_ids)

        elif mode == "evaluating":
            # Validation dataset
            self.dataset_val = Dataset()
            self.dataset_val.load_dataset('', self.DATASET_DIR + "/" + validation_dataset)
            self.dataset_val.prepare()

        elif mode == 'testing':
            print("Testing DATASET")
            # Testing dataset
            dataset_test = Dataset()
            dataset_test.load_dataset(testing_dataset[0], self.DATASET_DIR + "/" + testing_dataset[1])
            dataset_test.prepare()
            self.dataset_val = Dataset()
            self.dataset_val.load_dataset('', self.DATASET_DIR + "/" + validation_dataset)
            self.dataset_val.prepare()

    def create_model(self, mode='training', weights_path=None):
        if mode == 'training':
            # Create model in training mode (training)
            self.model = modellib.MaskRCNN(mode=mode, config=self.config, model_dir=self.MODEL_DIR)
            # Which weights to start with?
            if os.path.exists(self.ROOT_DIR + "/" + weights_path):
                model_path = self.ROOT_DIR + "/" + weights_path
            elif os.path.exists(self.WEIGTHS_DIR + "/" + weights_path):
                model_path = self.WEIGTHS_DIR + "/" + weights_path
            else:
                model_path = self.COCO_MODEL_PATH
            self.model.load_weights(model_path, by_name=True,
                                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

        else:
            from dataset import InferenceConfig
            inference_config = InferenceConfig()

            # Recreate the model in inference mode
            self.model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=self.MODEL_DIR)

            # Get path to saved weights
            # Either set a specific path or find last trained weights
            if os.path.exists(self.ROOT_DIR + "/" + weights_path):
                model_path = self.ROOT_DIR + "/" + weights_path
            else:
                model_path = self.WEIGTHS_DIR + "/" + weights_path
            # model_path = model.find_last()
            print(str(os.path.abspath(os.getcwd())))

            # Load trained weights
            print("Loading weights from ", model_path)
            self.model.load_weights(model_path, by_name=True,
                                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    def create_augmentation(self, visualize_augm=False):
        # The imgaug library is pretty flexible and make different types of augmentation possible.
        # The deterministic setting is used because any spatial changes to the image must also be
        # done to the mask. There are also some augmentors that are unsafe to apply. From the mrcnn
        # library:
        # Augmentors that are safe to apply to masks:
        # ["Sequential", "SomeOf", "OneOf", "Sometimes","Fliplr",
        # "Flipud", "CropAndPad", "Affine", "PiecewiseAffine"]
        # Affine, has settings that are unsafe, so always

        ia.seed(1)

        # http://imgaug.readthedocs.io/en/latest/source/augmenters.html#sequential
        self.seq_of_aug = iaa.Sequential([
            iaa.Crop(percent=(0, 0.1)),  # random crops

            # horizontally flip 50% of the images
            iaa.Fliplr(0.5),

            # Gaussian blur to 50% of the images
            # with random sigma between 0 and 0.5.
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),

            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.75, 1.5)),

            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),

            # Apply affine transformations to each image.
            # Scale/zoom them from 90% 5o 110%
            # Translate/move them, rotate them
            # Shear them slightly -2 to 2 degrees.
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-5, 5),
                shear=(-2, 2)
            )
        ], random_order=True)  # apply augmenters in random order

        if visualize_augm:
            # Some example augmentations using the seq defined above.
            image_id = np.random.choice(self.dataset_train.image_ids, 1)[0]
            image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(self.dataset_train, self.config, image_id,
                                                                              use_mini_mask=False)
            visualize.display_images([image], titles=['original'])
            image_list = []
            for i in range(15): image_aug = self.seq_of_aug.augment_image(image)
            image_list.append(image_aug)
            visualize.display_images(image_list, cols=5)

    def training(self, epochs=100, layers="heads"):
        # Train the head branches
        # Passing layers="heads" freezes all layers except the head
        # layers. You can also pass a regular expression to select
        # which layers to train bcy name pattern.
        self.model.keras_model.metrics_tensors = []
        self.model.train(self.dataset_train, self.dataset_val,
                         learning_rate=self.config.LEARNING_RATE,
                         epochs=epochs,
                         layers=layers,
                         augmentation=self.seq_of_aug
                         )

    def evaluation(self, number_of_images, confussion, directory):
        # Compute VOC-Style mAP @ IoU=0.5
        # Running on 'number_of_images' images. Increase for better accuracy.
        image_ids = np.random.choice(self.dataset_val.image_ids, number_of_images)

        APs = []

        gt_tot = np.array([])
        pred_tot = np.array([])

        if directory == None:
            directory = str(self.date)

        from dataset import InferenceConfig
        inference_config = InferenceConfig()

        savedir = self.model.log_dir

        keys = ["scores", "class_ids", "class_ids_gt", "bbox_gt", "mat"]
        DATOMAP = {}

        def createMap(key, values, res):
            for num, k in enumerate(key):
                if k not in res:
                    res[k] = []
                res[k].append(values[num])
            return res

        iterator = 0
        # Iteration all over the images randomly chosen
        for image_id in image_ids:
            iterator += 1
            mat = []
            # Load image and ground truth data
            print(str(iterator / len(image_ids) * 100) + '%:  processing image id: ' + str(image_id), flush=True)
            image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(self.dataset_val, inference_config, image_id, use_mini_mask=False)

            # Run object detection
            results = self.model.detect([image], verbose=0)
            r = results[0]

            gt, pred = utils.gt_pred_lists(gt_class_id, gt_bbox, r['class_ids'], r['rois'], iou_tresh=0.4)
            gt_tot = np.append(gt_tot, gt)
            pred_tot = np.append(pred_tot, pred)
            print("Ground truth: " + str(gt), flush=True)
            print("Prediction truth: " + str(pred), flush=True)
            # Compute AP
            AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"],
                                                                 r["class_ids"], r["scores"], r['masks'])
            APs.append(AP)

            new_direct = str(savedir) + "/" + str(image_id)
            self.create_directory(new_direct)

            '''
            print("The current len of the gt vect is : ", len(gt_tot))
            print("The current len of the pred vect is : ", len(pred_tot))
            print("The current precision is : ", precisions)
            print("The current recall is : ", recalls)
            print("The current overlaps is : ", overlaps)
            print("The current average precision : ", AP)
            '''
            print("The current average precision : ", np.mean(APs), flush=True)
            filename = str(image_id)
            # Display differences between both annotated and predicted
            visualize.display_differences(image, gt_bbox, gt_class_id, gt_mask, r['rois'], r['class_ids'], r['scores'],
                                          r['masks'],
                                          self.dataset_val.class_names, title="", ax=self.get_ax(size=20),
                                          show_mask=True, show_box=True,
                                          iou_threshold=0.5, score_threshold=0.5)

            plt.savefig(new_direct + "/differences.jpg")

            ma = len(results[0]['rois'])
            mi = list(map(str, range(ma)))

            # Display segmentation predicted for each image
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], self.dataset_val.class_names,
                                        r['scores'], ax=self.get_ax(size=20), captions=mi)
            plt.savefig(new_direct + "/segmentation.jpg")

            rois = results[0]['rois']
            for i, rect in enumerate(rois):
                y1, x1, y2, x2 = rect
                m = [x1, x2, y1, y2]
                mat.append(m)

            plt.close('all')
            values = [r['scores'], r['class_ids'], gt_class_id, gt_bbox, mat]
            new_dict = createMap(keys, values, {})
            pd.DataFrame(new_dict).to_csv(new_direct + "/results.csv", index=None)
            DATOMAP = createMap(["ImageID", "Precision", "Recall", "Overlaps", "AP"],
                                [image_id, np.mean(precisions), np.mean(recalls), np.mean(overlaps), AP], DATOMAP)

        pd.DataFrame(DATOMAP).to_csv(savedir + "/" + 'metrics.csv',
                                     header=["ImageID", "Precision", "Recall", "Overlaps", "AP"], index=None)
        np.savetxt(savedir + "/" + 'class_names.txt', self.dataset_val.class_names, fmt='%s')
        print("mAP: ", np.mean(APs), flush=True)

        # MATRIX OF CONFUSSION
        if confussion:
            confMatrix = confusionMatrix(self.model.log_dir)
            gt_pred_tot_json = {"gt_tot": gt_tot.astype(int), "pred_tot": pred_tot.astype(int)}
            df = pd.DataFrame(gt_pred_tot_json)
            df.to_json(os.path.join(savedir, 'gt_pred_test.json'))

            confMatrix.plot_confusion_matrix_from_data(gt_tot, pred_tot, columns=self.dataset_val.class_names,
                                                       annot=True,
                                                       cmap="Oranges",
                                                       fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[8, 8],
                                                       show_null_values=0, pred_val_axis='lin')

    def create_directory(self, name):
        """
        Create the directory where the output is going to be placed in
        :param name: name of the directory
        """
        if os.path.isdir(name) is False:
            os.mkdir(name)

    # Visualization
    def get_ax(self, rows=1, cols=1, size=8):
        """Return a Matplotlib Axes array to be used in
        all visualizations in the notebook. Provide a
        central point to control graph sizes.

        Change the default size attribute to control the size
        of rendered images
        """
        _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
        return ax

    def metadatos_extraction(self, directory="images/OPMM_Subasta_2020-09-01/", visualize_img=True, save=False,
                             llotja=False, new_path=None):
        from PIL import Image
        DATO = []
        DATOMAP = {}
        # self.create_directory(savedir)
        savedir = self.model.log_dir
        # DEBE ENTRAR EN LAS CARPETAS QUE SE ENCUENTRE CON EL NOMBRE OPPM_Subasta_"fecha"
        header2 = ["filename", "vta", "ord1", "emb", "pes", "caj", "fao", "score", "Number of Llampugues"]
        total_images = 0
        llampugues_img = 0
        no_llampugues_img = 0
        if llotja:
            if not directory.endswith("/"):
                directory += "/"
            path = directory
            self.INFERENCE_DIR = new_path
        else:
            if not directory.endswith("/"):
                directory += "/"
            path = self.ROOT_DIR + "/" + directory

        def getMetadatos(metadatos):
            dict = {}
            for metadato in metadatos.split("*"):
                if ":" in metadato:
                    key, value = metadato.split(":")
                    dict[key.replace('\x00', '')] = value.replace('\x00', '')
            return dict

        for root, dirs, files in os.walk(path):
            total_images = len(files)
            for filename in files:
                image = Image.open(root + filename)
                exifdata = image._getexif()
                if exifdata is not None:
                    for tag_id in exifdata:
                        data = exifdata.get(tag_id)
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                        metadatos = getMetadatos(data)
                        print(metadatos)
                        if "Ord" in metadatos:
                            criterion = ("FAO" in metadatos) and (metadatos["FAO"] == 'DOL') and (
                                        metadatos["Ord"] == '1') and (metadatos["Caj"] == '001')
                        else:
                            metadatos["Ord"] = 'None'
                            criterion = ("FAO" in metadatos) and (metadatos["FAO"] == 'DOL') and (
                                        metadatos["Caj"] == '001')
                        if criterion:
                            print(filename + " --> llampuga")
                            llampugues_img += 1
                            image = cv2.imread(root + filename)
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image, window, scale, padding, crop = utils.resize_image(
                                image,
                                min_dim=self.config.IMAGE_MIN_DIM,
                                min_scale=self.config.IMAGE_MIN_SCALE,
                                max_dim=self.config.IMAGE_MAX_DIM,
                                mode=self.config.IMAGE_RESIZE_MODE)

                            results = self.model.detect([image], verbose=1)

                            r = results[0]

                            ma = len(results[0]['rois'])
                            mi = list(map(str, range(ma)))
                            rois = results[0]['rois']
                            new_directory = self.getDirectory(filename)
                            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                                        self.dataset_val.class_names, r['scores'],
                                                        captions=mi, visualize_img=visualize_img, save=save,
                                                        output=new_directory + "/" + filename.split(".")[
                                                            0] + 'inference.jpg')

                            final = {}
                            final["scores"] = r['scores']
                            final["class_ids"] = r['class_ids']
                            mat = []
                            num_llamp = sum(list(map(lambda x: 1 if (x == 1) else 0, r['class_ids'])))
                            DAT = [filename, metadatos["Vta"], metadatos["Ord"], metadatos["Emb"], metadatos["Pes"],
                                   metadatos["Caj"], metadatos["FAO"], np.mean(r['scores']), num_llamp]

                            def createMap(key, values, res):
                                for num, k in enumerate(key):
                                    if k not in res:
                                        res[k] = []
                                    res[k].append(values[num])
                                return res

                            DATOMAP = createMap(header2, DAT, DATOMAP)
                            for i, r in enumerate(rois):
                                y1, x1, y2, x2 = r
                                m = [x1, x2, y1, y2]
                                mat.append(m)

                            final["bbox"] = mat

                            DATO.append(DAT)
                            if llotja is False:
                                pd.DataFrame(final).to_csv(savedir + "/" + filename + '_results.csv', index=None)
                        else:
                            print(filename + '--> no llampuga')
                            no_llampugues_img += 1
                    else:
                        print(filename + '--> no llampuga')
                        no_llampugues_img += 1
        if llotja is False:
            pd.DataFrame(DATOMAP).to_csv(savedir + "/" + 'DATOs.csv', header=header2, index=None)
        print("For total images: " + str(total_images) + ", we have detected " + str(
            llampugues_img) + " images of llampugues and " + str(no_llampugues_img) + " of another fishes", flush=True)

    def extract_date(self, filename):
        date = datetime.strptime(filename.split(".")[0], 'OPMM_Subasta_%Y-%m-%d_%H_%M_%S')
        return date

    def organizeFile(self, date):
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")

        self.create_directory(self.INFERENCE_DIR + "/" + year)
        self.create_directory(self.INFERENCE_DIR + "/" + year + "/" + month)
        self.create_directory(self.INFERENCE_DIR + "/" + year + "/" + month + "/" + day)

        return self.INFERENCE_DIR + "/" + year + "/" + month + "/" + day

    def getDirectory(self, filename):
        date = self.extract_date(filename)
        directory = self.organizeFile(date)
        return directory


class confusionMatrix():
    def __init__(self, directory):
        self.directory = directory

    def get_new_fig(self, fn, figsize=[9, 9]):
        """ Init graphics """
        fig1 = plt.figure(fn, figsize)
        ax1 = fig1.gca()  # Get Current Axis
        ax1.cla()  # clear existing plot
        return fig1, ax1

    #

    def configcell_text_and_colors(self, array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
        """
          config cell text and colors
          and return text elements to add and to dell
          @TODO: use fmt
        """
        import matplotlib.font_manager as fm

        text_add = [];
        text_del = [];
        cell_val = array_df[lin][col]
        tot_all = array_df[-1][-1]
        per = (float(cell_val) / tot_all) * 100
        curr_column = array_df[:, col]
        ccl = len(curr_column)

        # last line  and/or last column
        if (col == (ccl - 1)) or (lin == (ccl - 1)):
            # tots and percents
            if (cell_val != 0):
                if (col == ccl - 1) and (lin == ccl - 1):
                    tot_rig = 0
                    for i in range(array_df.shape[0] - 1):
                        tot_rig += array_df[i][i]
                    per_ok = (float(tot_rig) / cell_val) * 100
                elif (col == ccl - 1):
                    tot_rig = array_df[lin][lin]
                    per_ok = (float(tot_rig) / cell_val) * 100
                elif (lin == ccl - 1):
                    tot_rig = array_df[col][col]
                    per_ok = (float(tot_rig) / cell_val) * 100
                per_err = 100 - per_ok
            else:
                per_ok = per_err = 0

            per_ok_s = ['%.2f%%' % (per_ok), '100%'][per_ok == 100]

            # text to DEL
            text_del.append(oText)

            # text to ADD
            font_prop = fm.FontProperties(weight='bold', size=fz)
            text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
            lis_txt = ['%d' % (cell_val), per_ok_s, '%.2f%%' % (per_err)]
            lis_kwa = [text_kwargs]
            dic = text_kwargs.copy();
            dic['color'] = 'g';
            lis_kwa.append(dic);
            dic = text_kwargs.copy();
            dic['color'] = 'r';
            lis_kwa.append(dic);
            lis_pos = [(oText._x, oText._y - 0.3), (oText._x, oText._y), (oText._x, oText._y + 0.3)]
            for i in range(len(lis_txt)):
                newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
                # print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
                text_add.append(newText)
            # print '\n'

            # set background color for sum cells (last line and last column)
            carr = [0.27, 0.30, 0.27, 1.0]
            if (col == ccl - 1) and (lin == ccl - 1):
                carr = [0.17, 0.20, 0.17, 1.0]
            facecolors[posi] = carr

        else:
            if (per > 0):
                txt = '%s\n%.2f%%' % (cell_val, per)
            else:
                if (show_null_values == 0):
                    txt = ''
                elif (show_null_values == 1):
                    txt = '0'
                else:
                    txt = '0\n0.0%'
            oText.set_text(txt)

            # main diagonal
            if (col == lin):
                # set color of the textin the diagonal to white
                oText.set_color('w')
                # set background color in the diagonal to blue
                facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
            else:
                oText.set_color('r')

        return text_add, text_del

    #

    def insert_totals(self, df_cm):
        """ insert total column and line (the last ones) """
        sum_col = []
        for c in df_cm.columns:
            sum_col.append(df_cm[c].sum())
        sum_lin = []
        for item_line in df_cm.iterrows():
            sum_lin.append(item_line[1].sum())
        df_cm['sum_lin'] = sum_lin
        sum_col.append(np.sum(sum_lin))
        df_cm.loc['sum_col'] = sum_col
        # print ('\ndf_cm:\n', df_cm, '\n\b\n')

    #

    def pretty_plot_confusion_matrix(self, df_cm, annot=True, cmap="Oranges", fmt='.2f', fz=11,
                                     lw=0.5, cbar=False, figsize=[8, 8], show_null_values=0, pred_val_axis='y'):
        """
          print conf matrix with default layout (like matlab)
          params:
            df_cm          dataframe (pandas) without totals
            annot          print text in each cell
            cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
            fz             fontsize
            lw             linewidth
            pred_val_axis  where to show the prediction values (x or y axis)
                            'col' or 'x': show predicted values in columns (x axis) instead lines
                            'lin' or 'y': show predicted values in lines   (y axis)
        """
        from matplotlib.collections import QuadMesh
        import seaborn as sn

        if (pred_val_axis in ('col', 'x')):
            xlbl = 'Predicted'
            ylbl = 'Actual'
        else:
            xlbl = 'Actual'
            ylbl = 'Predicted'
            df_cm = df_cm.T

        # create "Total" column
        self.insert_totals(df_cm)

        # this is for print allways in the same window
        fig, ax1 = self.get_new_fig('Conf matrix default', figsize)

        # thanks for seaborn
        ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                        cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

        # set ticklabels rotation
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=25, fontsize=10)

        # Turn off all the ticks
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False

        # face colors list
        quadmesh = ax.findobj(QuadMesh)[0]
        facecolors = quadmesh.get_facecolors()

        # iter in text elements
        array_df = np.array(df_cm.to_records(index=False).tolist())
        text_add = [];
        text_del = [];
        posi = -1  # from left to right, bottom to top.
        for t in ax.collections[0].axes.texts:  # ax.texts:
            pos = np.array(t.get_position()) - [0.5, 0.5]
            lin = int(pos[1]);
            col = int(pos[0]);
            posi += 1
            # print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

            # set text
            txt_res = self.configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt,
                                                      show_null_values)

            text_add.extend(txt_res[0])
            text_del.extend(txt_res[1])

        # remove the old ones
        for item in text_del:
            item.remove()
        # append the new ones
        for item in text_add:
            ax.text(item['x'], item['y'], item['text'], **item['kw'])

        # titles and legends
        ax.set_title('Confusion matrix')
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        plt.tight_layout()  # set layout slim
        plt.savefig(self.directory + "/confusion_matrix.jpg")
        plt.show()

    #

    def plot_confusion_matrix_from_data(self, y_test, predictions, columns=None, annot=True, cmap="Oranges",
                                        fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[8, 8], show_null_values=0,
                                        pred_val_axis='lin'):
        """
            plot confusion matrix function with y_test (actual values) and predictions (predic),
            whitout a confusion matrix yet
        """
        from sklearn.metrics import confusion_matrix
        from pandas import DataFrame

        # data
        if (not columns):
            # labels axis integer:
            ##columns = range(1, len(np.unique(y_test))+1)
            # labels axis string:
            from string import ascii_uppercase
            columns = ['class %s' % (i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]

        confm = confusion_matrix(y_test, predictions)
        cmap = 'Oranges';
        fz = 11;
        figsize = [9, 9];
        show_null_values = 2
        df_cm = DataFrame(confm, index=columns, columns=columns)
        self.pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values,
                                          pred_val_axis=pred_val_axis)
    #


class automatiscm():
    def initialize(self, mode, weight_path, training_dataset, validation_dataset, testing_dataset, visualize_images,
                   llotja=False):
        self.demo = Demo_toni()
        self.demo.initialization(weight_path=weight_path)
        self.demo.configuration()
        self.demo.initialize_dataset(mode=mode, training_dataset=training_dataset,
                                     validation_dataset=validation_dataset, testing_dataset=testing_dataset,
                                     visualize_images=visualize_images, llotja=llotja)
        self.demo.create_model(mode=mode, weights_path=weight_path)

    def training(self, epochs, layers, vis_augmentation):
        self.demo.create_augmentation(visualize_augm=vis_augmentation)
        self.demo.training(epochs=epochs, layers=layers)

    def evaluation(self, number_of_images, confussion, directory):
        self.demo.evaluation(number_of_images=number_of_images, confussion=confussion, directory=directory)

    def testing(self, directory="images/OPMM_Subasta_2020-09-01/", visualize=True, save=False, llotja=False,
                new_path=None):
        self.demo.metadatos_extraction(directory=directory, visualize_img=visualize, save=save, llotja=llotja,
                                       new_path=new_path)

    def parse_arguments(self):
        import argparse
        # Import arguments into the python script
        parser = argparse.ArgumentParser("IMEDEA: Project Deep Ecomar")
        parser.add_argument("--directory", type=str, default="images/OPMM_Subasta_2020-09-01/",
                            help="Path to the directory of images to be tested")
        parser.add_argument('--not_display', action="store_false", default=True,
                            help='Introduce the argumnent to not display the processing of the images step by step')
        parser.add_argument('--save', action="store_true", default=False,
                            help='Introduce the argument to save the processed images')
        parser.add_argument('--notconfussion', action="store_false", default=True,
                            help='Whether to show the mattrix of confussion or not')
        parser.add_argument('--mode', type=str, default="testing",
                            help='Introduce the aim of the execution: training, evaluating or testing')
        parser.add_argument('--weights_path', type=str, default="weights/mask_rcnn_toni_0045.h5",
                            help='Introduce the path to the weigths')
        parser.add_argument('--epochs', type=int, default=100, help='Introduce the number of epochs for the training')
        parser.add_argument('--image_num', type=int, default=30,
                            help='Introduce the number of images you want to process')
        parser.add_argument('--layers', type=str, default="all",
                            help='Introduce the layers tou want to train: either "all" or "heads"')
        parser.add_argument('--train_dataset', type=str, default="train.json",
                            help='Introduce the path to the training dataset')
        parser.add_argument('--val_dataset', type=str, default="/toni_test/ann/coco_test.json",
                            help='Introduce the path to the validation dataset')
        parser.add_argument('--test_dataset_images', type=str, default='',
                            help='Introduce the path to the testing dataste image directory')
        parser.add_argument('--test_dataset_file', type=str, default='/test.json',
                            help='Introduce the path to the testing dataset json file')

        args = parser.parse_args()

        # Parse arguments into variables
        return args

    def main(self):
        print("[INFO] Starting ...", flush=True)
        args = self.parse_arguments()
        mode = args.mode
        print("[INFO] Initializing ...", flush=True)

        self.initialize(mode=mode, weight_path=args.weights_path, training_dataset=args.train_dataset,
                        validation_dataset=args.val_dataset,
                        testing_dataset=[args.test_dataset_images, args.test_dataset_file], visualize_images=False)

        if mode == "training":
            print("[INFO] Training ...", flush=True)
            self.training(epochs=args.epochs, layers=args.layers, vis_augmentation=args.not_display)

        elif mode == "testing":
            print("[INFO] Testing ...", flush=True)
            self.testing(directory=args.directory, visualize=args.not_display, save=args.save)

        elif mode == "evaluating":
            print("[INFO] Evaluating ...", flush=True)
            self.evaluation(directory=args.directory, confussion=args.notconfussion, number_of_images=args.image_num)

        else:
            print("Error while choosing the Mode", flush=True)


if __name__ == "__main__":
    llampuga = automatiscm()
    llampuga.main()
