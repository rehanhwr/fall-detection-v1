from collections import OrderedDict
import torch
from torch import nn
from torch import optim
from utils import parse_args
from utils import plot_loss_acc
from utils import save_points
from utils import set_parameter_requires_grad
from utils import load_split_train_test
from utils import write_training_result
from utils import get_data_to_print
from utils import accuracy_topk
import time
import copy

root_path = "data_dummy/"
data_path_train = root_path + 'train/'
data_path_validation = root_path + 'validation/'
input_size = 224
model_name = 'squeezenet1_0'
save_path = "./saved_model/model_ft"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device used: ', device)


def main(args):
  data_dir = args.dataset_name
  num_epochs = args.epoch
  feature_extract = args.feature_extract
  batch_size = args.batch_size
  validation_size = args.validation_size
  num_classes = args.classes

  print('Dataset dir: ', data_dir)

  if feature_extract:
    model_ft = torch.load(save_path)
  else:
    model_ft = torch.hub.load('pytorch/vision:v0.4.2', 'squeezenet1_0', pretrained=True)

  model_ft = set_parameter_requires_grad(model_ft, feature_extract)
  model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
  model_ft.num_classes = num_classes
  # print(model)

  # Create training and validation dataloaders
  dataloaders_dict, sz_dict = load_split_train_test(data_dir, batch_size, input_size, validation_size)
  # Send the model to GPU (Optimizer)
  model_ft = model_ft.to(device)

  # Gather the parameters to be optimized/updated in this run. If we are
  #  finetuning we will be updating all parameters. However, if we are
  #  doing feature extract method, we will only update the parameters
  #  that we have just initialized, i.e. the parameters with requires_grad
  #  is True.
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

  # Observe that all parameters are being optimized
  optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

  # Setup the loss fxn
  criterion = nn.CrossEntropyLoss()

  # Train and evaluate
  model_ft, train_loss, val_acc, batch_lost_acc = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, sz_dict=sz_dict)
  save_points(model_ft, save_path)
  print()
  print('Train Loss: {}'.format(train_loss))
  print('Val Acc: {}'.format(val_acc))
  # plot_loss_acc(train_loss, val_acc, num_epochs)



def train_model(model, dataloaders, criterion, optimizer, num_epochs, sz_dict):
  since = time.time()

  val_acc_history = []
  train_loss_history = []

  batch_val_loss_history = []
  batch_val_acc_history = []

  batch_train_loss_history = []
  batch_train_acc_history =[]

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(num_epochs):
    since_epoch = time.time()

    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()  # Set model to training mode
      else:
        model.eval()   # Set model to evaluate mode

      running_loss = 0.0
      running_corrects = 0

      print(phase)
      print('=' * 20)
      # Iterate over data.

      batch_cnt = 0
      for inputs, labels in dataloaders[phase]:
        since_batch = time.time()

        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            # Get model outputs and calculate loss
          outputs = model(inputs)
          loss = criterion(outputs, labels)

          _, preds = torch.max(outputs, 1)

          if phase == 'val':
            probs = torch.exp(outputs)
            acc_topk_res = accuracy_topk(probs, labels.data, (1,5))
            top1_acc = acc_topk_res[0].item()
            top5_acc = acc_topk_res[1].item()
            print("Top-1 Acc: {}, Top-5 Acc: {}".format(top1_acc, top5_acc))
            write_training_result("{},{}\n".format(top1_acc, top5_acc), "./top1_top5_acc.csv")

          # backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        batch_loss = loss.item()
        batch_acc = (torch.sum(preds == labels.data)*1.0)/inputs.size(0)
        
        if phase == 'train':
          batch_train_loss_history.append(batch_loss)
          batch_train_acc_history.append(batch_acc)
        else:
          batch_val_loss_history.append(batch_loss)
          batch_val_acc_history.append(batch_acc)

        data_to_print, location = get_data_to_print(epoch, phase, batch_loss, batch_acc)
        write_training_result(data_to_print, location)


        print('BATCH {} Loss: {:.4f} Acc: {:.4f}'.format(phase, batch_loss, batch_acc))
        time_elapsed_batch = time.time() - since_batch
        print('One batch completed in {:.0f}m {:.0f}s'.format(time_elapsed_batch // 60, time_elapsed_batch % 60))
        print()

        SAVE_MODEL_PATH_BATCH = './saved_model/batch'+ str(batch_cnt) + '_epoch' + str(epoch) + "_saved_model.pth"
        save_points(SAVE_MODEL_PATH_BATCH, epoch, model, optimizer, batch_cnt, phase, loss)

        batch_cnt+=1

      epoch_loss = running_loss / sz_dict[phase]
      epoch_acc = running_corrects.double() / sz_dict[phase]

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

      # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
      if phase == 'val':
        val_acc_history.append(epoch_acc.item())
      else:
        train_loss_history.append(epoch_loss)

    time_elapsed_epoch = time.time() - since_epoch
    print('One epoch completed in {:.0f}m {:.0f}s'.format(time_elapsed_epoch // 60, time_elapsed_epoch % 60))
    print()

  time_elapsed = time.time() - since
  print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  SAVE_MODEL_PATH = './saved_model/epoch' + str(epoch) + "_saved_model.pth"
  save_points(SAVE_MODEL_PATH, epoch, model, optimizer)

  # load best model weights
  model.load_state_dict(best_model_wts)

  batch_lost_acc = {
    'train_loss': batch_train_loss_history,
    'train_acc': batch_train_acc_history,
    'val_loss': batch_val_loss_history,
    'val_acc': batch_val_acc_history
  }
  return model, train_loss_history, val_acc_history, batch_lost_acc

if __name__ == '__main__':
  args = parse_args()
  main(args)
