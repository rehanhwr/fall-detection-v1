import argparse
from collections import OrderedDict
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

data_path_train = 'data/train/'
data_path_validation = 'data/validation/'

def main(args):
  model = torch.hub.load('pytorch/vision:v0.4.2', 'squeezenet1_0', pretrained=True)
  model.eval()

  classifier =nn.Sequential(OrderedDict([('fc1', nn.Linear(13, 256)),
                            ('relu', nn.ReLU()), 
                            ('dropout', nn.Dropout(p=0.337)),
                            ('fc2', nn.Linear(256, 3)),
                            ('output', nn.LogSoftmax(dim=1))
                          ]))
  model.classifier = classifier

  criteria = nn.CrossEntropyLoss()

  #Initialize training params  
  #freeze gradient parameters in pretrained model
  for param in model.parameters():
    param.require_grad = False
  #train and validate
  epochs = 10  
  epoch = 0
  #send model to GPU
  if args.gpu:
    model.to('cuda')
  
  # TODO: load using script
  trainLoader = load_dataset(data_path_train)
  validLoader = load_dataset(data_path_validation)

  for e in range(epochs):
    epoch +=1
    print(epoch)
    with torch.set_grad_enabled(True):
      epoch_train_loss, epoch_train_acc = train(model,trainLoader, criteria, args.gpu)
    print("Epoch: {} Train Loss : {:.4f}  Train Accuracy: {:.4f}".format(epoch,epoch_train_loss,epoch_train_acc))
    with torch.no_grad():
      epoch_val_loss, epoch_val_acc = validation(model, validLoader, criteria, args.gpu)
    print("Epoch: {} Validation Loss : {:.4f}  Validation Accuracy {:.4f}".format(epoch,epoch_val_loss,epoch_val_acc))



def train (model, loader, criterion, gpu):
  model.train()
  current_loss = 0
  current_correct = 0
  optimizer = optim.SGD(model.parameters(), lr = 0.005, momentum = 0.5)

  for train, y_train in iter(loader):
    if gpu:
      train, y_train = train.to('cuda'), y_train.to('cuda')
    optimizer.zero_grad()
    output = model.forward(train)
    _, preds = torch.max(output,1)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    current_loss += loss.item()*train.size(0)
    current_correct += torch.sum(preds == y_train.data)
  epoch_loss = current_loss / len(trainLoader.dataset)
  epoch_acc = current_correct.double() / len(trainLoader.dataset)
      
  return epoch_loss, epoch_acc



def validation (model, loader, criterion, gpu):
  model.eval()
  valid_loss = 0
  valid_correct = 0
  for valid, y_valid in iter(loader):
    if gpu:
      valid, y_valid = valid.to('cuda'), y_valid.to('cuda')
    output = model.forward(valid)
    valid_loss += criterion(output, y_valid).item()*valid.size(0)
    equal = (output.max(dim=1)[1] == y_valid.data)
    valid_correct += torch.sum(equal)#type(torch.FloatTensor)
  
  epoch_loss = valid_loss / len(validLoader.dataset)
  epoch_acc = valid_correct.double() / len(validLoader.dataset)
  
  return epoch_loss, epoch_acc


def parse_args():
  parser = argparse.ArgumentParser(description='PyTorch Example')
  parser.add_argument('--disable-cuda', action='store_true',
                      help='Disable CUDA')
  parser.add_argument('-gpu', action='store_false',
                      help='Disable CUDA')
  args = parser.parse_args()
  args.device = None
  args.gpu = None
  if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    args.gpu = True
  else:
    args.device = torch.device('cpu')
    args.gpu = False
  
  return args


def load_dataset(data_path):
  dataset = datasets.ImageFolder(
    root=data_path,
    transform=transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
  )
  data_loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=0,
    shuffle=True
  )

  return data_loader



if __name__ == '__main__':
  args = parse_args()
  main(args)