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

    parser.add_argument('path_to_img', type=str, help='path to the image to classify')
    parser.add_argument('path_to_checkpoint', type=str, help='path to the model checkpoint')
    # options
    parser.add_argument('--top_k', type = int, default=3, help = 'return top k most likely classes') 
    parser.add_argument('--category_names', type = str, default='cat_to_name.json', help = 'mapping of categories to real names') 
    parser.add_argument('--gpu', action='store_true', default = False, help = 'to enable gpu') 

    return parser.parse_args()


def load_checkpoint(checkpoint_filename):
    checkpoint = torch.load(checkpoint_filename)
    model=checkpoint['model']
    optimizer=checkpoint['optimizer']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model, optimizer


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    
    # resize
    px_resize = 256
    img_resize = image.resize((px_resize,px_resize))
    
    # crop
    px_crop = 224
    left= 0.5*(px_resize-px_crop)
    upper= left
    right= 0.5*(px_resize+px_crop)
    lower= right
    img_crop = img_resize.crop(box=(left, upper, right, lower))
    
    # normalize
    img_norm = np.array(img_crop)/255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_processed = (img_norm - mean) / std
    
    return img_processed.transpose((2,0,1))


def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    
    # no dropout
    model.eval()
    
    # GPU or CPU
    if gpu and torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
        
    # preprocess image
    img = Image.open(image_path)
    img_processed = process_image(img)
    
    inputs = torch.from_numpy(img_processed)
    if gpu and torch.cuda.is_available():
        inputs = Variable(inputs.float().cuda())
    else:
        inputs = Variable(inputs)
    inputs = inputs.unsqueeze(0)
    
    # Forward pass
    outputs = model.forward(inputs)

    # get probabilities and top classes
    ps = torch.exp(outputs).data.topk(topk)
    probs, indices = ps[0].cpu().numpy()[0], ps[1].cpu().numpy()[0]
    
    # map idx to classes
    idx_to_class = {model.class_to_idx[key]: key for key in model.class_to_idx}
    classes = []
    for idx in indices:
        classes.append(idx_to_class[idx])
    
    return probs, classes


def main():

    # get arguments from command line
    args = get_input_args()

    # load model
    model, optimizer = load_checkpoint(args.path_to_checkpoint)

    # predict image classes
    probs, classes = predict(args.path_to_img, model, args.top_k, args.gpu)

    # get category labels
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    labels = []
    for c in classes:
        labels.append(cat_to_name[c])

    # print results
    print('probs: ', probs)
    print('labels: ', labels)



if __name__ == "__main__":
    main()