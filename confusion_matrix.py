from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import torch
from torch import nn
from utils import set_parameter_requires_grad
from utils import load_split_train_test
from utils import parse_args
import matplotlib.pyplot as plt
from torch import optim
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from utils import write_training_result


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 11
LOAD_PATH = './saved_model/' + 'batch294_epoch0_saved_model.pth'
data_dir = "/gs/hs0/tga-isshiki-lab/rehan/dataset/"
# data_dir = "./data/classes/"
input_size = 224
validation_size = 0.3

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()



def plot_cfm(cfm, classes):
  array = cfm
  df_cm = pd.DataFrame(array, index = classes, columns = classes)
  plt.figure(figsize = (10,7))
  sn.heatmap(df_cm, annot=True)



def load_inference_model(args):
  batch_size = args.batch_size
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
      # Get model outputs and calculate loss
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
  np.set_printoptions(precision=2)

  # Plot non-normalized confusion matrix
  plt.figure()
  torch.save({
    "cnf_matrix": cnf_matrix,
    }, "./cnf_matrix.pth")
  print("SELESAII")
  # plot_confusion_matrix(cnf_matrix, classes=classes, title='Confusion matrix, without normalization')
  plot_cfm(cnf_matrix, classes)


if __name__ == '__main__':
  args = parse_args()
  main(args)
