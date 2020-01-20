import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np


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

if __name__ == '__main__':
  main()