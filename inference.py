# import the necessary packages
import argparse
import shutil

from demo_llampuga import automatiscm
import datetime
import py7zr
import glob
import os

__author__ = 'EsteveValls'


class Inference():
    def __init__(self):
        args = self.parserArguments()
        self.demoLlampuga = automatiscm()
        self.mode = "Testing"
        self.weigth_path = args.weights
        self.testing_dataset = "testing/test.json"
        self.directory = args.directory
        self.output_path = args.output
        self.base_results = "/home/amaya/FOTOPEIX/MASK_RCNN/ESTEVE/inference"
        self.base_input="/extusers/llotja/"
        print("[INFO] Initializing ...", flush=True)
        self.demoLlampuga.initialize(mode=self.mode, training_dataset=None, validation_dataset="val.json", weight_path=self.weigth_path, testing_dataset=[self.testing_dataset, self.testing_dataset], visualize_images=False, llotja=True)


    def parserArguments(self):
        # Import arguments into the python script
        parser = argparse.ArgumentParser("Inference")
        parser.add_argument("-i", "--directory", type=str, default="~/FOTOPEIX/MASK_RCNN/ESTEVE/inference", help="Path to the directory of the images to be analyzed")
        parser.add_argument("--weights", type=str, default="mask_rcnn_llampuga_0068.h5", help="Path to the weights to use for the inference")
        parser.add_argument("--output", type=str, default="inference", help="Path to the output directory to store the images")

        args = parser.parse_args()
        return args

    def inference(self):
        print("[INFO] Starting ...", flush=True)
        print("[INFO] Inference ...", flush=True)
        self.demoLlampuga.testing(directory=self.directory, visualize=False, save=True, llotja=True, new_path=self.base_results)

    def getDate(self, name):
        if "/" in name:
            checker = name.split("/")
            name = checker[-1]

        date = datetime.datetime.strptime(name, 'OPMM_Subasta_%Y-%m-%d.7z') #OPMM_Subasta_YYYY-mm-dd.7z
        self.date = date.strftime("%Y_%m_%d")
        return date

    def filter_days(self,date):
        current_year = datetime.date.today().year
        start_date = datetime.datetime(current_year, 8, 25)
        end_date = datetime.datetime(current_year, 12, 25)
        return ((start_date <= date) and (date <= end_date))

    def extract_directory(self, auction_path):
        archive = py7zr.SevenZipFile(self.base_input+"/"+auction_path, mode='r')
        archive.extractall(path=self.base_results)
        archive.close()

    def processAuction(self, auction_path):
        date = self.getDate(auction_path)
        if self.filter_days(date):
            print("[INFO] Carregant Subasta del dia "+ str(self.date)+" ...")
            self.extract_directory(auction_path)
            print("[INFO] Imatges de la llotja extretes correctament.")
            directory = auction_path.split(".")[0]
            self.directory = self.base_results+"/"+directory
            self.inference()
            self.remove_directory()
        else:
            print("[INFO] Subasta del dia "+ str(self.date)+" no està en període de llampugues")

    def read_summary(self):
        # open results file
        try:
            with open(self.base_results+'/Summary.txt', "r+") as fil:
                # first we read the file
                processedAuction = fil.readlines()

        except Exception:
            print('File {} does not exist. We proceed on creating it'.format(self.base_results+'/Summary.txt'))
            f = open(self.base_results+'/Summary.txt', "x+")
            processedAuction = []
        return [str(x).replace("\n",'') for x in processedAuction]

    def write_summary(self, name):
        # open results file
        try:
            with open(self.base_results+'/Summary.txt', "a") as fil:
                fil.write(name+"\n")

        except IOError:
            print('File {} does not exist. We proceed on creating it'.format('out/Summary.txt'))
            f = open(self.base_results+'/Summary.txt', "x+")
            f.write(name+"\n")

    def remove_directory(self):
        try:
            shutil.rmtree(self.directory)
        except:
            print('Error while deleting directory: ' +self.directory)

    def processDirectory(self):
        # grab the paths to the input images and initialize our images list
        print("[INFO] Carregant Subastes...")

        txtfiles = []
        for file in glob.glob(self.base_input+"/*.7z"):
            txtfiles.append(os.path.basename(file))
        auctions_path = sorted(txtfiles)
        #print("Auctions path:" + str(auctions_path))
        processed_Auctions = self.read_summary()
        print("Processed auctions:" + str(processed_Auctions))

        filtered_path = [x for x in auctions_path if x not in processed_Auctions]

        length = len(filtered_path)
        print("Processing " +str(length)+ " auctions")

        # loop over the image paths, load each one, and add them to our
        # images to stich list
        for auction_path in filtered_path:
                self.processAuction(auction_path)
                self.write_summary(auction_path)

if __name__ == "__main__":
    inference = Inference()
    inference.processDirectory()
