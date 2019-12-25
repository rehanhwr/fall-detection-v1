import csv
from PIL import Image

ROOT_DIR = "./dataset/"
PREFIX_ACT_DIR = "adl/"
SUFFIX_ACT_DIR = "-cam0-rgb"
EXTENSION = ".png"
PREFIX_IDX = 0
FRAME_IDX = 1
LABEL_IDX = 2

def leading_zero(curr_str):
	while (len(curr_str)) < 3:
		curr_str = "0" + curr_str
	return curr_str

csv_path = "./dataset/urfall-cam0-adls.csv"
with open(csv_path) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        # print(row)
        print(row)
        ACT_DIR_NAME = row[PREFIX_IDX] + SUFFIX_ACT_DIR
        image_path = ROOT_DIR + PREFIX_ACT_DIR + ACT_DIR_NAME + "/" + ACT_DIR_NAME + "-" + leading_zero(row[FRAME_IDX]) + EXTENSION
        img = Image.open(image_path)

        # new_class_name = 
        new_path = "./data/classes/" + row[LABEL_IDX] + "/" + ACT_DIR_NAME + "-" + leading_zero(row[FRAME_IDX]) + EXTENSION
        img = img.save(new_path)
        # print(row[0],row[1],row[2],)

