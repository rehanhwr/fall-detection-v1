import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import os
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime
import random




def main():
  pass


def seed_torch(seed=1029):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
  torch.backends.cudnn.benchmark = False
  # torch.backends.cudnn.deterministic = True


def parse_args():
  parser = argparse.ArgumentParser(description='PyTorch Example')
  parser.add_argument("-dn", "--dataset-name", type=str, default='./data_dummy/classes/')
  parser.add_argument("-e", "--epoch", type=int, default=1)
  parser.add_argument("-f", "--feature-extract", action='store_true')
  parser.add_argument("-b", "--batch-size", type=int, default=300) # default cuda
  parser.add_argument("-v", "--validation-size", type=float, default=.3) # default cuda
  parser.add_argument("-s", "--save_path", type=str)
  parser.add_argument("-p", "--proxy", type=str)
  parser.add_argument("-mt", "--max-try", type=int)
  parser.add_argument("-c", "--classes", type=int)
  parser.add_argument("-r", "--resume-training", type=int, default=0)
  parser.add_argument("-l", "--load-path", type=str)

  args = parser.parse_args()
  return args


def plot_loss_acc(losses, accuracies, num_epochs):
  # lhist = []
  # ahist = []

  # lhist = [l.cpu().numpy() for l in losses]
  # ahist = [a.cpu().numpy() for a in accuracies]

  plt.title("Training Loss and Validation Accuracy")
  plt.xlabel("Epochs")
  plt.ylabel("Loss/Accuracy")
  plt.plot(range(1,num_epochs+1),accuracies,label="Val Acc")
  plt.plot(range(1,num_epochs+1),losses,label="Train Loss")
  plt.ylim((0,1.))
  plt.xticks(np.arange(1, num_epochs+1, 1.0))
  plt.legend()
  plt.show()


def save_points(PATH, epoch, model, optimizer, batch="-1", phase=None, loss=None):
  torch.save({
    'batch': batch,
    'phase': phase,
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
    }, PATH)


def set_parameter_requires_grad(model, feature_extracting):
  if feature_extracting:
    for param in model.parameters():
      param.requires_grad = False

  return model


# DEPRECATED
def load_dataset(data_dir, batch_size, input_size):
  # Data augmentation and normalization for training
  # Just normalization for validation
  data_transforms = {
    'train': transforms.Compose([
      transforms.RandomResizedCrop(input_size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
      transforms.Resize(input_size),
      transforms.CenterCrop(input_size),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
  }

  print("Initializing Datasets and Dataloaders...")

  # Create training and validation datasets
  image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
  dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

  return dataloaders_dict


def load_split_train_test(datadir, batch_size=64, input_size=224, valid_size = .15):
  print("Initializing Datasets and Dataloaders...")

  data_transforms = {
    'train': transforms.Compose([
      transforms.Resize(input_size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
      transforms.Resize(input_size),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
  }

  train_data = datasets.ImageFolder(datadir, transform=data_transforms['train'])
  test_data = datasets.ImageFolder(datadir, transform=data_transforms['val'])

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(valid_size * num_train))
  print("Total data: ", num_train)
  np.random.seed(7)
  np.random.shuffle(indices)

  train_idx, test_idx = indices[split:], indices[:split]
  train_cnt = len(train_idx)
  val_cnt = len(test_idx)
  print("Number of data Training: ", train_cnt)
  print("Number of data Validation: ", val_cnt)

  train_sampler = SubsetRandomSampler(train_idx)
  test_sampler = SubsetRandomSampler(test_idx)

  trainloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, worker_init_fn=np.random.seed(12))
  testloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size, worker_init_fn=np.random.seed(12))

  dataloaders_dict = {
    'train': trainloader,
    'val': testloader
  }

  sz_dict = {
    'train': train_cnt,
    'val': val_cnt
  }

  return dataloaders_dict, sz_dict


def write_training_result(data, to_path):
  f = open(to_path, 'a')
  # today = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
  # f.write('Result on {}\n'.format(today))
  # f.write(str(data) + str('\n\n'))
  f.write(str(data))
  f.close()


def get_data_to_print(epoch, phase, loss, acc):
  location = ''
  if phase == 'train':
    location = './train_result_batch.csv'
  else:
    location = './val_result_batch.csv'
  data = "{},{},{},{}\n".format(epoch, phase, loss, acc)
  return data, location


def accuracy_topk(output, target, topk=(1,)):
  res = []
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
  return res


if __name__ == '__main__':
  main()
