import os
import random
import math
from PIL import Image

ROOT_DIR = "./data/"
CLASS_DIR = "classes/"
TRAIN_TYPE = "train/"
VALIDATION_TYPE = "validation/"
LABEL = ["-1", "0", "1"]
SLASH_SYMBOL = "/"

def main():
	for label in LABEL:
		path = ROOT_DIR + CLASS_DIR + label + SLASH_SYMBOL
		entries = os.listdir(path)

		random.seed(7)
		sample_k = math.ceil(len(entries) * 0.15)
		validation = random.sample(population=entries, k=sample_k)

		for entry in entries:
			img_path = path + entry
			img = Image.open(img_path)
		
			new_img_path = ROOT_DIR
			if entry in validation:
				new_img_path += VALIDATION_TYPE
			else:
				new_img_path += TRAIN_TYPE
			new_img_path += label + SLASH_SYMBOL + entry

			img = img.save(new_img_path)

			print(label + " -- " + new_img_path)

if __name__ == "__main__":
	main()