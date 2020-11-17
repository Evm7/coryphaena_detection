import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

import imgaug as ia
from imgaug import augmenters as iaa

import mrcnn.utils as utils
import mrcnn.model as modellib
import mrcnn.visualize as visualize
import inspect
import mrcnn.config as config

from datetime import *
from dataset import DatasetConfig
import pandas as pd


class Demo_LLampuga():

    def __init__(self, weight_path):
        self.class_names = ["BG", "llampuga", "ticket"]
        print("Initialization of root directories scheme:")
        # Root directory of the project
        self.ROOT_DIR = os.path.abspath("")
        print("\t-ROOT DIR:" + self.ROOT_DIR)

        # Import Mask RCNN
        sys.path.append("")  # To find local version of the library
        os.chdir("mrcnn")

        module = inspect.getmodule(config)
        module_path = os.path.dirname(module.__file__)
        print("\t-MODULE PATH:" + module_path)

        # Directory to save logs and trained model
        self.MODEL_DIR = os.path.join(self.ROOT_DIR, "logs")
        self.create_directory(self.MODEL_DIR)
        print("\t-MODEL_DIR PATH:" + self.MODEL_DIR)

        self.WEIGTHS_DIR = os.path.join(self.ROOT_DIR, "weights")
        self.create_directory(self.WEIGTHS_DIR)

        self.INFERENCE_DIR = os.path.join(self.ROOT_DIR, "inference")
        self.create_directory(self.INFERENCE_DIR)
        print("\t-INFERENCE_DIR PATH:" + self.INFERENCE_DIR)

        self.LOGS_DIR = os.path.join(self.ROOT_DIR, "logs")
        self.create_directory(self.LOGS_DIR)
        print("\t-LOGS_DIR PATH:" + self.LOGS_DIR)

        # Local path to trained weights file
        self.COCO_MODEL_PATH = os.path.join(self.ROOT_DIR, "weights/mask_rcnn_coco.h5")

        if weight_path == None:
            # Download COCO trained weights from Releases if needed
            utils.download_trained_weights(self.COCO_MODEL_PATH)

    def configuration(self):
        self.config = DatasetConfig()
        self.config.display()

        self.date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        try:
            self.config.save(self.MODEL_DIR, str(self.date))
        except FileNotFoundError:
            print("File /logs/" + str(self.date) + " does not exist. We proceed to create so")
            f = open(self.MODEL_DIR + "/Config_" + str(self.date) + ".txt", "x+")
            f.write("Configurations")
            f.close()
            self.config.save(self.MODEL_DIR, str(self.date))

    def initialize_dataset(self, mode, train_annotations, train_images, val_annotations,val_images, visualize_images=False, llotja=False):
        if llotja or mode == "inference":
            return

        from dataset import Dataset

        # Validation dataset
        self.dataset_val = Dataset()
        self.dataset_val.load_dataset(val_images, val_annotations)
        self.dataset_val.prepare()


        if mode == 'training':
            # Training dataset
            self.dataset_train = Dataset()
            self.dataset_train.load_dataset(train_images,train_annotations)
            self.dataset_train.prepare()

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


    def create_model(self, mode, weights_path=None):
        # Weights instantiation
        if os.path.exists(self.ROOT_DIR + "/" + weights_path):
            model_path = self.ROOT_DIR + "/" + weights_path
        elif os.path.exists(self.WEIGTHS_DIR + "/" + weights_path):
            model_path = self.WEIGTHS_DIR + "/" + weights_path
        else:
            model_path = self.COCO_MODEL_PATH

        print("[INFO].. Using weights " +str(model_path))

        if mode == 'training':
            # Create model in training mode (training)
            self.model = modellib.MaskRCNN(mode=mode, config=self.config, model_dir=self.MODEL_DIR)
            self.model.load_weights(model_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

        else:
            from dataset import InferenceConfig
            inference_config = InferenceConfig()
            # Recreate the model in inference mode
            self.model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=self.MODEL_DIR)
            self.model.load_weights(model_path, by_name=True)

    def create_augmentation(self, visualize_augm=False):
        # The imgaug library is pretty flexible and make different types of augmentation possible.
        # The deterministic setting is used because any spatial changes to the image must also be
        # done to the mask. There are also some augmentors that are unsafe to apply. From the mrcnn
        # library:
        # Augmentors that are safe to apply to masks:
        # ["Sequential", "SomeOf", "OneOf", "Sometimes","Fliplr",
        # "Flipud", "CropAndPad", "Affine", "PiecewiseAffine"]
        # Affine, has settings that are unsafe, so always

        print("[INFO].. Using augmentation for the Training")

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

    def evaluation(self, number_of_images, confussion):
        # Running on 'number_of_images' images. Increase for better accuracy.
        image_ids = np.random.choice(self.dataset_val.image_ids, number_of_images)

        APs = []

        gt_tot = np.array([])
        pred_tot = np.array([])

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
            print("\t-Ground truth: " + str(gt), flush=True)
            print("\t-Prediction truth: " + str(pred), flush=True)
            # Compute AP
            AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"],
                                                                 r["class_ids"], r["scores"], r['masks'])
            APs.append(AP)

            new_direct = str(savedir) + "/" + str(image_id)
            self.create_directory(new_direct)

            print("The current average precision : ", np.mean(APs), flush=True)
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
            from confusion_matrix import confusionMatrix
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

    def metadatos_extraction(self, directory, visualize_img=True, save=False, llotja=False, new_path=None):
        from PIL import Image
        DATO = []
        DATOMAP = {}
        savedir = self.model.log_dir
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
                        if not llotja:
                            print(metadatos)
                        if "Ord" in metadatos:
                            criterion = ("FAO" in metadatos) and (metadatos["FAO"] == 'DOL') and (
                                        metadatos["Ord"] == '1') and (metadatos["Caj"] == '001')
                        else:
                            metadatos["Ord"] = 'None'
                            criterion = ("FAO" in metadatos) and (metadatos["FAO"] == 'DOL') and (
                                        metadatos["Caj"] == '001')
                        if criterion:
                            if not llotja:
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
                                                        self.class_names, r['scores'],
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
                            if not llotja:
                                print(filename + '--> no llampuga')
                            no_llampugues_img += 1
                    else:
                        if not llotja:
                            print(filename + '--> no llampuga')
                        no_llampugues_img += 1
        if llampugues_img>0:
            if llotja is False:
                pd.DataFrame(DATOMAP).to_csv(savedir + "/" + 'DATOs.csv', header=header2, index=None)
            else:
                direct = self.getDirectory_for_llotja(filename)
                pd.DataFrame(DATOMAP).to_csv(direct + "/" + 'DATOs.csv', header=header2, index=None)

        print("For total images: " + str(total_images) + ", we have detected " + str(llampugues_img) + " images of llampugues and " + str(no_llampugues_img) + " of another fishes", flush=True)

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

    def getDirectory_for_llotja(self, filename):
        date = self.extract_date(filename)
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")

        self.create_directory(self.INFERENCE_DIR + "/datos")
        self.create_directory(self.INFERENCE_DIR + "/datos/" + year)
        self.create_directory(self.INFERENCE_DIR + "/datos/" + year + "/" + month)
        self.create_directory(self.INFERENCE_DIR + "/datos/" + year + "/" + month + "/" + day)

        return self.INFERENCE_DIR+ "/datos/" + year + "/" + month + "/" + day


class automatiscm():
    def initialize(self, mode, weight_path, train_annotations,train_images, val_annotations, val_images, visualize_images, llotja=False):
        self.demo = Demo_LLampuga(weight_path=weight_path)
        self.demo.configuration()
        self.demo.initialize_dataset(mode=mode, train_annotations=train_annotations,train_images=train_images, val_annotations=val_annotations ,val_images=val_images, visualize_images=visualize_images, llotja=llotja)
        self.demo.create_model(mode=mode, weights_path=weight_path)

    def training(self, epochs, layers, vis_augmentation):
        self.demo.create_augmentation(visualize_augm=vis_augmentation)
        self.demo.training(epochs=epochs, layers=layers)

    def evaluation(self, number_of_images, confusion):
        self.demo.evaluation(number_of_images=number_of_images, confussion=confusion)

    def inference(self, directory, visualize=True, save=False, llotja=False, new_path=None):
        self.demo.metadatos_extraction(directory=directory, visualize_img=visualize, save=save, llotja=llotja, new_path=new_path)

    def parse_arguments(self):
        import argparse
        # Import arguments into the python script
        parser = argparse.ArgumentParser("IMEDEA: Project Deep Ecomar")
        parser.add_argument("--directory", type=str, default="", help="Path to the directory of images to be tested")
        parser.add_argument('--not_display', action="store_false", default=True, help='Introduce the argument to not display the processing of the images step by step')
        parser.add_argument('--save', action="store_true", default=False, help='Introduce the argument to save the processed images')
        parser.add_argument('--not_confusion', action="store_false", default=True, help='Whether to show the mattrix of confussion or not')
        parser.add_argument('--mode', type=str, default="", help='Introduce the aim of the execution: training, evaluating or inference')
        parser.add_argument('--weights_path', type=str, default="", help='Introduce the path to the weigths')
        parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for the training')
        parser.add_argument('--image_num', type=int, default=30, help='Number of images to process in evaluation')
        parser.add_argument('--layers', type=str, default="all", help='Introduce the layers tou want to train: either "all" or "heads"')
        parser.add_argument('--train_annotations', type=str, default="", help='Introduce the path to the training dataset annotation file')
        parser.add_argument('--val_annotations', type=str, default="", help='Introduce the path to the validation dataset annotation file')
        parser.add_argument('--train_images', type=str, default="", help='Introduce the path to the training image directory')
        parser.add_argument('--val_images', type=str, default="", help='Introduce the path to the validation image directory')
        args = parser.parse_args()

        # Parse arguments into variables
        return args

    def checkArguments(self, args):
        if args.mode == "training":
            if args.train_annotations == "":
                print("[ERROR]... Training annotation file not introduced. Use: --train_annotations [PATH] !")
                sys.exit(0)

        elif args.mode == "inference":
            if args.directory == "":
                print("[ERROR]... Directory to infer not introduced. Use: --directory [PATH] !")
                sys.exit(0)

        elif args.mode == "evaluating":
            if args.val_annotations == "":
                print("[ERROR]... Training dataset not introduced. Use: --val_annotations [PATH] !")
                sys.exit(0)
        else:
            print("[ERROR]... Mode not correctly selected. Please select --mode [{training, inference, evaluating}]", flush=True)
            sys.exit(0)

        if args.weights_path == "":
            print("[ERROR]... Weights path not introduced. Use: --weights_path [PATH] !")
            sys.exit(0)

    def main(self):
        print("[INFO] Starting ...", flush=True)

        # Retrieve argument information
        args = self.parse_arguments()
        self.checkArguments(args)

        # Deciding mode
        mode = args.mode
        print("[INFO] Initializing ...", flush=True)

        self.initialize(mode=mode, weight_path=args.weights_path, train_annotations=args.train_annotations, train_images=args.train_images,
                        val_annotations=args.val_annotations, val_images=args.val_images, visualize_images=False)

        if mode == "training":
            print("[INFO] Training ...", flush=True)
            self.training(epochs=args.epochs, layers=args.layers, vis_augmentation=args.not_display)

        elif mode == "inference":
            print("[INFO] Inference ...", flush=True)
            self.inference(directory=args.directory, visualize=args.not_display, save=args.save)

        elif mode == "evaluating":
            print("[INFO] Evaluating ...", flush=True)
            self.evaluation(confusion=args.not_confusion, number_of_images=args.image_num)

        else:
            print("Error while choosing the Mode", flush=True)


if __name__ == "__main__":
    llampuga = automatiscm()
    llampuga.main()
