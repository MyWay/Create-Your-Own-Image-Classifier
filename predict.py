#!/usr/bin/env python3
""" train.py
predict.py Predict the flower class from an image.
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import json
from PIL import Image
import numpy as np
import predict_args

def main():
    # load the cli args
    parser = predict_args.get_args()
    cli_args = parser.parse_args()

    use_cuda = cli_args.use_gpu

    # load categories
    with open(cli_args.categories_json, 'r') as f:
        cat_to_name = json.load(f)

    model_transfer = load_from_checkpoint(cli_args.checkpoint_file)
    if use_cuda:
        model_transfer.cuda()
    top_probs, top_classes = predict(cli_args.path_to_image, model_transfer, use_cuda, topk=cli_args.top_k)
    
    label = top_classes[0]
    prob = top_probs[0]

    print(f'Input\n---------------------------------')

    print(f'Image\t\t:\t{cli_args.path_to_image}')
    print(f'Model\t\t:\t{cli_args.checkpoint_file}')

    print(f'\nPrediction\n---------------------------------')

    print(f'Flower\t\t:\t{cat_to_name[label]}')
    print(f'Label\t\t:\t{label}')
    print(f'Probability\t:\t{prob*100:.2f}%')

    print(f'\nTop K\n---------------------------------')

    for i in range(len(top_probs)):
        print(f"{cat_to_name[top_classes[i]]:<25} {top_probs[i]*100:.2f}%")
    
def load_from_checkpoint(path):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model_transfer = models.__dict__['alexnet'](pretrained=True)
    model_transfer.classifier = checkpoint['classifier']
    model_transfer.load_state_dict(checkpoint['state_dict'])
    model_transfer.class_to_idx = checkpoint['class_to_idx']
    
    for param in model_transfer.parameters(): 
        param.requires_grad = False
        
    return model_transfer

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])
    
    data = Image.open(image).convert('RGB')
    image = transform(data)
       
    return image
    
def predict(image_path, model, use_cuda=False, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    
    image = process_image(image_path)
    image = image.unsqueeze(0)
    
    if use_cuda:
        model.cuda()
        image.cuda()
    else:
        model.cpu()
    
    # No grad
    with torch.no_grad():
        output = model.forward(image)
        # AlexNet is returning scores, we need to softmax to get probabilities
        
        probabilities = torch.exp(output)
        
        top_probs, top_labels = torch.topk(probabilities, topk)
        
        top_probs = top_probs.numpy()
        top_labels = top_labels.numpy() 

        top_probs = top_probs.tolist()[0]
        top_labels = top_labels.tolist()[0]


        mapping = {val: key for key, val in model.class_to_idx.items() }

        top_labels = [mapping [item] for item in top_labels]
        top_labels = np.array(top_labels)

    return top_probs, top_labels
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)