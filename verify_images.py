from PIL import Image
import os
from utils import write_training_result

activities = []
for i in range(11):
  activities.append('Activity' + str(i+1))

def main(args):
  ROOT = args.save_path

  cnt_valid = 0
  cnt_corrupt = 0
  for act in activities:
    path = ROOT + act + '/'
    entries = os.listdir(path)

    for entry in entries:
      img_path = path + entry
      img = Image.open(img_path)
      try:
        img.verify()
        # print('Valid image')
        cnt_valid+=1
      except Exception:
        print('Invalid image: ', img_path)
        write_training_result(img_path, "corrupted_images.txt")
        cnt_corrupt+=1
      img.close()

  print(cnt_valid, ' valid images.')
  print(cnt_corrupt, ' corrupted images.')
  print('Total images: ', cnt_valid+cnt_corrupt)


if __name__ == '__main__':
  args = parse_args()
  main(args)
