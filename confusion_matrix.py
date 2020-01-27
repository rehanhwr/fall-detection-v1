import torch
from torch import nn
from torch import optim
from sklearn.metrics import confusion_matrix
from utils import set_parameter_requires_grad
from utils import load_split_train_test
from utils import parse_args
from utils import write_training_result
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import itertools
import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 11
# data_dir = "./data/classes/"
input_size = 224

def load_inference_model(args):
  # LOAD_PATH = './saved_model/batch294_epoch0_saved_model.pth'
  # data_dir = "/gs/hs0/tga-isshiki-lab/rehan/dataset/"
  data_dir = args.dataset_name
  LOAD_PATH = args.load_path
  batch_size = args.batch_size
  validation_size = args.validation_size

  feature_extract = False

  model_ft = torch.hub.load('pytorch/vision:v0.4.2', 'squeezenet1_0', pretrained=True)
  model_ft = set_parameter_requires_grad(model_ft, feature_extract)
  model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
  model_ft.num_classes = num_classes

  params_to_update = model_ft.parameters()
  print("Params to learn:")
  if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
      if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)
  else:
    for name,param in model_ft.named_parameters():
      if param.requires_grad == True:
        print("\t",name)

  optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

  checkpoint = torch.load(LOAD_PATH)
  model_ft.load_state_dict(checkpoint['model_state_dict'])
  optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])

  dataloaders_dict, sz_dict = load_split_train_test(data_dir, batch_size, input_size, validation_size)

  model_ft.eval()
  criterion = nn.CrossEntropyLoss()

  inputs, labels = next(iter(dataloaders_dict['val']))
  inputs = inputs.to(device)
  labels = labels.to(device)

  with torch.set_grad_enabled(False):
    outputs = model_ft(inputs)
    loss = criterion(outputs, labels)

    _, preds = torch.max(outputs, 1)
    return preds, labels.data


def main(args):
  y_pred, y_true = load_inference_model(args)
  print("y_pred", y_pred)
  print("y_true", y_true)

  classes=[x for x in range(num_classes)]
  cnf_matrix = confusion_matrix(y_true, y_pred, labels=classes)

  torch.save({
    "cnf_matrix": cnf_matrix,
  }, "./cnf_matrix.pth")
  print("SELESAII")


if __name__ == '__main__':
  args = parse_args()
  main(args)
