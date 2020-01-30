import gdown
import os
from utils import parse_args
from dataset_fetcher import find_last_downloaded_data
from dataset_fetcher import get_next_index_data_name
from dataset_fetcher import dataset_name
from dataset_fetcher import idx_to_data_name


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

def write_downloaded_data(sub, act, tri, cam, is_na=False):
  string = '{},{},{},{}'.format(sub, act, tri, cam)
  if is_na:
    string += "-- NA"
  string += "\n"
  print('Write downloaded data: ', string)
  f = open('extracted_data.csv', 'a')
  f.write(string)
  f.close()


def main(args):
  # save_path = /gs/hs0/tga-isshiki-lab/rehan
  ROOT = args.save_path
  save_path = ROOT

  lsub, lact, ltri, lcam = find_last_downloaded_data('extracted_data.csv')
  if lsub != '':
    next_idx = get_next_index_data_name(lsub, lact, ltri, lcam)

  while next_idx < len(dataset_name):
    sub, act, tri, cam = idx_to_data_name(next_idx)
    print()
    print('=' * 20)
    print('Preparing to extract next data ...', sub, act, tri, cam)

    file_name = sub + act + tri + cam + ".zip"
    file_path = ROOT + '/new_dataset/' + sub + '/' + act + '/' + tri + '/' + cam + '/' + file_name

    save_path = ROOT + '/dataset/' + act + '/'
    if os.path.exists(file_path):
      if not os.path.exists(save_path):
        os.makedirs(save_path)

      print('Extracting ', file_name)
      gdown.extractall(file_path, save_path)
      write_downloaded_data(sub,act,tri,cam)
      print()
    else:
      print("!!!!!")
      print('Data Not Available: ', file_name)
      write_downloaded_data(sub,act,tri,cam, is_na=True)
      print()
    next_idx+=1

  print("=== Finished")


    

if __name__ == '__main__':
  args = parse_args()
  main(args)
