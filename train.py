# Imports
import torch, torchvision, numpy as np, matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import time
import copy
from PIL import Image

import argparse
import json


def get_input_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str, help='path to the folder containing the images')
    # options
    parser.add_argument('--save_dir', type = str, default='', help = 'path to the folder for saving the model checkpoint')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'architecture of the CNN')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'learning rate for training the model')
    parser.add_argument('--hidden_units', type = int, default = 512, help = 'number of hidden units for the classifier')
    parser.add_argument('--epochs', type = int, default = 20, help = 'number of epochs for training the model')
    parser.add_argument('--gpu', action='store_true', default = False, help = 'to enable gpu')

    return parser.parse_args()


def train_model(model, criterion, optimizer, num_epochs, gpu, train_dataloader, valid_dataloader):

    # use dropout
    model.train()

    # GPU or CPU
    if gpu and torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    
    total_step = len(train_dataloader)
    
    # Iterate over each epoch
    for epoch in range(num_epochs):
        train_loss = 0
        
        # Iterate over each image
        for step, (inputs, labels) in enumerate(train_dataloader):
            if gpu and torch.cuda.is_available():
                inputs = Variable(inputs.float().cuda())
                labels = Variable(labels.long().cuda()) 
            else:
                inputs = Variable(inputs)
                labels = Variable(labels) 
                    
            # Forward pass
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            
            # Backward prop
            loss.backward()
            optimizer.step() 
            train_loss += loss.item()

            # Calculate validation loss
            if (step+1) % 50 == 0:
                valid_loss, valid_acc = valid_model(model, criterion, gpu, valid_dataloader)
                print ('Epoch [{}/{}] '.format(epoch+1, num_epochs),
                       'Step [{}/{}] '.format(step+1, total_step),
                       'Train Loss: {:.3f}'.format(train_loss),
                       'Valid Loss: {:.3f}'.format(valid_loss),
                       'Valid Accuracy: {:.3f}'.format(valid_acc))


def valid_model(model, criterion, gpu, valid_dataloader):
    
    # no dropout
    model.eval()
    
    valid_loss = 0
    valid_acc = 0
    
    n_valid_imgs = len(valid_dataloader)
    
    # Iterate over each image
    for inputs, labels in iter(valid_dataloader):
        if gpu and torch.cuda.is_available():
            inputs = Variable(inputs.float().cuda())
            labels = Variable(labels.long().cuda()) 
        else:
            inputs = Variable(inputs)
            labels = Variable(labels) 
            
        # Forward pass
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)

        # update loss
        valid_loss += loss.item()
        
        # update accuracy
        ps = torch.exp(outputs).data 
        equality = (labels.data == ps.max(1)[1])
        valid_acc += equality.type_as(torch.FloatTensor()).mean()
        
    return valid_loss/n_valid_imgs, valid_acc/n_valid_imgs


def main():

    # get arguments from command line
    args = get_input_args()


    # load data
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    train_transform = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

    valid_transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

    test_transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, valid_transform)
    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    # Using the image datasets and the tranforms, define the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)


    # Load the pre-trained network
    model = getattr(models, args.arch)(pretrained=True)

    # freeze weights of pre-trained network
    for param in model.parameters():
        param.requires_grad = False

    # get label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # get number of input units for the classifier
    input_units = model.classifier[0].in_features

    # Define a new, untrained feed-forward network
    classifier = nn.Sequential(
        nn.Linear(in_features=input_units, out_features=args.hidden_units, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=args.hidden_units, out_features=len(cat_to_name), bias=True)
    )
    model.classifier = classifier

    # get Hyper-parameters from command line args
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    criterion = nn.CrossEntropyLoss()
    gpu = args.gpu
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)

    # train model
    train_model(model, criterion, optimizer, num_epochs, gpu, train_dataloader, valid_dataloader)

    # save model
    checkpoint_filename = args.save_dir + 'model_checkpoint.pth'
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {'model': model,
                  'optimizer': optimizer,
                  'class_to_idx': model.class_to_idx,
                  'model_state_dict': model.state_dict(), 
                  'optimizer_state_dict': optimizer.state_dict(), 
                }
    torch.save(checkpoint, checkpoint_filename)



if __name__ == "__main__":
    main()