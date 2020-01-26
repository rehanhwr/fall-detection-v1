import gdown
import os
from utils import parse_args


subjects = []
for i in range(17):
  subjects.append('Subject' + str(i+1))

activities = []
for i in range(11):
  activities.append('Activity' + str(i+1))

trials = []
for i in range(3):
  trials.append('Trial' + str(i+1))

cameras = []
for i in range(2):
  cameras.append('Camera' + str(i+1))

def write_downloaded_data(sub, act, tri, cam):
  string = '{},{},{},{}\n'.format(sub, act, tri, cam)
  print('Write downloaded data: ', string)
  f = open('extracted_data.csv', 'a')
  f.write(string)
  f.close()


def main(args):
  # save_path = /gs/hs0/tga-isshiki-lab/rehan
  ROOT = args.save_path
  save_path = ROOT

  for sub in subjects:
    for act in activities:
      for tri in trials:
        for cam in cameras:
          file_name = sub + act + tri + cam + ".zip"
          file_path = ROOT + '/new_dataset/' + sub + '/' + act + '/' + tri + '/' + cam + '/' + file_name
          # extract

          save_path = ROOT + '/dataset/' + act + '/'
          
          if os.path.exists(file_path):
            if not os.path.exists(save_path):
              os.makedirs(save_path)
            print('Extracting ', file_name)
            gdown.extractall(file_path, save_path)
            write_downloaded_data(sub,act,tri,cam)
            print()
  print("=== Finished")


    

if __name__ == '__main__':
  args = parse_args()
  main(args)
