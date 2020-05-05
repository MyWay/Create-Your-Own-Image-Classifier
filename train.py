#!/usr/bin/env python3
""" train.py
train.py train the model for the flower dataset.
"""

# Imports here
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict
import torch.optim as optim
import numpy as np
from PIL import Image
import random
import json
import train_args

def main():
    parser = train_args.get_args()
    cli_args = parser.parse_args()

    if cli_args.arch != 'alexnet':
        print('Currently, we support only AlexNet.')
        exit(1)
        
    
    use_cuda = False
    epochs = cli_args.epochs
    checkpoint_name = 'checkpoint.pt'
        
    if cli_args.save_dir:
        save_dir = cli_args.save_dir
        
    if cli_args.save_name:
        save_name = cli_args.save_name
        
    if save_dir and save_name:
        checkpoint_name = f'{cli_args.save_dir}/{cli_args.save_name}.pt'
        
    # check if CUDA is available and if we want to use it
    if cli_args.use_gpu and torch.cuda.is_available():
        use_cuda = True
    else:
        print("GPU is not available. Using CPU.")
        
    hidden_units = cli_args.hidden_units
    
    # check for data directory
    if not os.path.isdir(cli_args.data_directory):
        print(f'Data directory {cli_args.data_directory} was not found.')
        exit(1)

    # check for save directory
    if not os.path.isdir(cli_args.save_dir):
        print(f'Directory {cli_args.save_dir} does not exist. Creating...')
        os.makedirs(cli_args.save_dir)
    
    # load the directory
    train_dir = cli_args.data_directory
    valid_dir = 'flowers/valid'

    train_transform = transforms.Compose([transforms.RandomRotation(15),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])
    
    valid_transform = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])

    data_transforms = {'train': train_transform, 'valid': valid_transform}
    
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    valid_dataset = ImageFolder(valid_dir, transform=valid_transform)

    image_datasets = {'train': train_dataset, 'valid': valid_dataset}
    
    batch_size=20
    num_workers=0

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                           num_workers=num_workers, shuffle=False)

    dataloaders = {'train': train_loader, 'valid': valid_loader}
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    model_transfer, criterion_transfer, optimizer_transfer = get_model(use_cuda=use_cuda, hidden_units=hidden_units)
    model_transfer = train(dataloaders, model_transfer, optimizer_transfer, criterion_transfer, checkpoint_name, epochs, use_cuda, train_dataset)

def get_model(dropout=0.5, use_cuda=False, hidden_units=None):
    flower_categories=102
    
    model_transfer = models.alexnet(pretrained=True)

    for param in model_transfer.features.parameters():
        param.requires_grad = False
    
    # Use AlexNet features and weights for fast converging
    fc1_weights = model_transfer.classifier[1].weight
    fc2_weights = model_transfer.classifier[4].weight
    
    classifier = None
    hidden_units = hidden_units[0]
    if hidden_units:
        classifier = nn.Sequential(OrderedDict([
            ('dropout1', nn.Dropout(dropout)),
            ('fc1', nn.Linear(model_transfer.classifier[1].in_features, hidden_units)),
            ('relu1', nn.ReLU(inplace=True)),
            ('dropout2', nn.Dropout(dropout)),
            ('fc2', nn.Linear(hidden_units, hidden_units)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(hidden_units, flower_categories)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    else:
        classifier = nn.Sequential(OrderedDict([
            ('dropout1', nn.Dropout(dropout)),
            ('fc1', nn.Linear(model_transfer.classifier[1].in_features, model_transfer.classifier[1].out_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('dropout2', nn.Dropout(dropout)),
            ('fc2', nn.Linear(model_transfer.classifier[4].in_features, model_transfer.classifier[4].out_features)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(model_transfer.classifier[6].in_features, flower_categories)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    
    model_transfer.classifier = classifier
    
    if not hidden_units:
        # Restoring AlexNet weights
        model_transfer.classifier[1].weight = fc1_weights
        model_transfer.classifier[4].weight = fc2_weights
    
    if use_cuda:
        model_transfer = model_transfer.cuda()
        
    # Define loss and optimizer
    criterion_transfer = nn.NLLLoss()
    optimizer_transfer = optim.SGD(model_transfer.classifier.parameters(), lr=0.001)
        
    return model_transfer, criterion_transfer, optimizer_transfer

def train(loaders, model, optimizer, criterion, checkpoint_name, n_epochs=11, use_cuda=False, train_dataset=None):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
                model = model.cuda()
                
            # initialize weights to zero
            optimizer.zero_grad()
                
            # get the loss
            output = model(data)
            loss = criterion(output, target)
            
            # backward propagation
            loss.backward()
            
            # update weights
            optimizer.step()
            
            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            
            # update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            model.class_to_idx = train_dataset.class_to_idx
            torch.save({'state_dict': model.state_dict(), 
                        'class_to_idx': model.class_to_idx,
                        'classifier': model.classifier,
                       'optimizer_dict': optimizer.state_dict(),
                       'n_epochs': n_epochs},
                        checkpoint_name)
            valid_loss_min = valid_loss
            
    # return trained model
    return model

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    