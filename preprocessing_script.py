import os
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

def separate_adl_fall_to_classes():
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


# (takes time) : 3 classes --> 1 dataset (each image has class name) --> random split 85-15 --> save to traditional structure 
# (alternatives) : 1 dataset (already combined) --> random split --> save in python strcuture or tensor!

def entries_to_name(entries_name):
  return entries_name.split('.')[0]


# working in data, not dummy data
def renaming_images_with_class_name():
  root_dir = "./data/"
  phases = ['train/', 'val/']
  classes = ["-1", "0", "1"]

  move = False
  for phase in phases:
    for clas in classes:
      if phase == 'train/' and clas != '1':
        continue

      image_path = root_dir + phase + clas + "/"
      entries = os.listdir(image_path)

      for entry in entries:
        entry_path = image_path + entry
        img = Image.open(entry_path)

        if move:
          new_path = "./data/all/" + entries_to_name(entry) + '_' + clas + EXTENSION 
          img = img.save(new_path)
          print('Saved to ', new_path)

        if entry == 'adl-23-cam0-rgb-156.png':
          move = True


if __name__ == '__main__':
  renaming_images_with_class_name()
