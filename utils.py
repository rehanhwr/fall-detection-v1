import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import os


def main():
  pass

def parse_args():
  parser = argparse.ArgumentParser(description='PyTorch Example')
  parser.add_argument("-dn", "--dataset-name", type=str, default='data_dummy/')
  parser.add_argument("-e", "--epoch", type=int, default=1)
  parser.add_argument("-f", "--feature-extract", action='store_true')

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


def save_points(model, path):
  torch.save(model, path)


def set_parameter_requires_grad(model, feature_extracting):
  if feature_extracting:
    for param in model.parameters():
      param.requires_grad = False

  return model


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


if __name__ == '__main__':
  main()