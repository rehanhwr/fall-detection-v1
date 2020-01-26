from bs4 import BeautifulSoup
import csv
import gdown
import os
from utils import parse_args


# generate data constant names
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


def find_last_downloaded_data(path='./downloaded_data.csv'):
  with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    csv_reader = list(csv_reader)
    row_len = len(csv_reader)

    
    subject = ''
    activity = ''
    trial = ''
    camera = ''

    if row_len > 1:
      row = csv_reader[-1]
      subject = row[0]
      activity = row[1]
      trial = row[2]
      camera = row[3]
  return subject, activity, trial, camera


def generate_all_dataset_names():
  dataset_name = []
  for sub in subjects:
    for act in activities:
      for tri in trials:
        for cam in cameras:
          dataset_name.append(sub+'_'+act+'_'+tri+'_'+cam)
  return dataset_name


dataset_name = generate_all_dataset_names()


def get_next_index_data_name(sub, act, tri, cam):
  last_data_name = sub+'_'+act+'_'+tri+'_'+cam
  next_idx = ''
  for idx, name in enumerate(dataset_name, 1):
    if name == last_data_name:
      next_idx = idx
      break

  return next_idx


def idx_to_data_name(idx):
  data = dataset_name[idx].split('_')
  return data[0], data[1], data[2], data[3] 


def write_downloaded_data(sub, act, tri, cam):
  string = '{},{},{},{}\n'.format(sub, act, tri, cam)
  print('Write downloaded data: ', string)
  f = open('downloaded_data.csv', 'a')
  f.write(string)
  f.close()


def main(args):
  SAVE_PATH = args.save_path
  PROXY = args.proxy
  UP_DATASET_URL = 'https://sites.google.com/up.edu.mx/har-up/'
  # current loacation
  print('Parsing dataset page ...')
  page = open('UP_dataset.html')
  soup = BeautifulSoup(page.read(), 'html.parser')

  next_idx = 0
  print('Find last downloaded data ...')
  lsub, lact, ltri, lcam = find_last_downloaded_data()
  if lsub != '':
    next_idx = get_next_index_data_name(lsub, lact, ltri, lcam)
  
  while next_idx < len(dataset_name):
    nsub, nact, ntri, ncam = idx_to_data_name(next_idx)
    print("\n\n")
    print('=' * 20)
    print('Preparing to download next data ...')
    div_data = soup.find('div', attrs={'id': nsub+nact+ntri})
    anchor = div_data.find('a', href=True, text=ncam)
    
    url = anchor['href']
    output_path = SAVE_PATH + '/new_dataset/' + nsub + '/' + nact + '/' + ntri + '/' + ncam + '/' 
    
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    
    output_path += nsub+nact+ntri+ncam + '.zip'
    
    print('Downloading data to: ', output_path)
    gdown.download(url, output_path, quiet=False, proxy=PROXY)

    write_downloaded_data(nsub, nact, ntri, ncam)
    next_idx+=1
  
  print('All data downloaded ...')



if __name__ == '__main__':
  args = parse_args()
  main(args)
