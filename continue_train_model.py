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


def main():
  pass


'''
resume_dict = {
  'epoch': checkpoint['epoch'],
  'loss': checkpoint['loss'],
  'batch_cnt': checkpoint['batch'],
  'phase': checkpoint['phase']
}
'''
def continue_train_model(model, dataloaders, criterion, optimizer, num_epochs, sz_dict, resume_dict):
  since = time.time()

  val_acc_history = []
  train_loss_history = []

  batch_val_loss_history = []
  batch_val_acc_history = []

  batch_train_loss_history = []
  batch_train_acc_history =[]

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(int(resume_dict['epoch']), num_epochs):
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
  main()

