import torch

import torch.nn as nn

import torch.optim as optim

import torchvision as tv

import torchvision.models as models

import time

import copy

import pandas as pd

import os

from PIL import Image

# Dataset class used to iterate through data

class ImageAndWeatherDataset(torch.utils.data.Dataset):
    
    def __init__(self, csv_file, img_dir, transforms = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label = self.csv_file.iloc[idx,1]
        
        img_path = os.path.join(self.img_dir,
                                self.csv_file.iloc[idx, 0])
        
        image = Image.open(img_path)
        
        image = self.transforms(image)
        
        weather_data = self.csv_file.iloc[idx,2:].to_numpy().astype('float')
        
        return (image, weather_data, label)

def get_dataloader(input_img_size,csv_file,DatasetKlass,img_dir,
                    batch_size):
        
    # only normalize images in training, validation, and testing sets
    
    img_transforms = tv.transforms.Compose([
                     tv.transforms.Resize(256),
                     tv.transforms.CenterCrop(input_img_size),
                     tv.transforms.ToTensor(),
                     tv.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])])
        
    # create training, validation, and testing datasets
    
    dataset = DatasetKlass(csv_file,img_dir,img_transforms)
            
    # create iterators for training, validation, and testing datasets
    
    dataloader = torch.utils.data.DataLoader(dataset = dataset,
                                             batch_size = batch_size,
                                             shuffle = True,
                                             num_workers = 0)
            
    return dataloader

def train_and_val_model(model,optimizer,scheduler,loss_func,train_dataloader,
                        val_dataloader,device):
    
    # TRAINING PHASE ----------------------------------------------------------
    
    # put model in training mode
    
    model.train()
    
    # initialize train_loss_accum that will be used to accumulate loss values
    # across an entire epoch for each batch and to compute the epoch loss
    
    train_loss_accum = 0
    
    # initialize train_num_corr_pred that will be used to accumulate the number
    # of correct predictions across an entire epoch for batch and to compute
    # the epoch training accuracy
    
    train_num_corr_pred = 0    
    
    # compute the total number of samples in the training set
    
    train_set_size = len(train_dataloader.dataset)
    
    # get the batch size for the training set
    
    train_batch_size = train_dataloader.batch_size
    
    dtype = (torch.cuda.FloatTensor if torch.cuda.is_available()
             else torch.FloatTensor)
    
    # iterate over all batches in training set
    
    for (images, weather_data, labels) in train_dataloader:
        
        # put batch of images onto GPU if available
        
        images = images.to(device)
        
        # put batch of weather data onto GPU if available
        
        weather_data = weather_data.to(device).type(dtype)
        
        # normalize to avoid exploding and vanishing gradients
        
        weather_data = nn.functional.normalize(weather_data,p=2,dim=0)
        
        # put batch of labels onto GPU if available
        
        labels = labels.to(device)
        
        # zero parameter gradients
        
        optimizer.zero_grad()
        
        # gradient tracking is NOT enabled since images and labels both have
        # requires_grad = False
        
        with torch.set_grad_enabled(True):
        
            # forward propagation. No need to put outputs onto GPU because
            # images are already on GPU. This means tha outputs will automatically
            # be on GPU
        
            outputs = model(images,weather_data)
            
            # compute class predictions
            
            preds = torch.argmax(outputs, dim = 1)
            
            # compute loss. No need to put loss onto GPU since both outputs and
            # labels are on GPU. This means that loss will automatically be put
            # onto GPU
            
            loss = loss_func(outputs,labels)
            
            # backward propagation
            
            loss.backward()
            
            # update weights
            
            optimizer.step()
        
        # return loss as scalar after performing backward propagation
        
        loss = loss.item()
        
        # accumulate loss for each batch
        
        train_loss_accum = train_loss_accum + (loss*train_batch_size)
        
        # accumulate number of correct predictions for each batch
        
        train_num_corr_pred = train_num_corr_pred + torch.sum(preds == labels)
        
    # compute the average training loss across the epoch
    
    epoch_train_loss = train_loss_accum / train_set_size
    
    # compute the training accuracy for the epoch
    
    epoch_train_acc = train_num_corr_pred.item() / train_set_size
        
    # VALIDATION PHASE --------------------------------------------------------
    
    # put model in inference mode
    
    model.eval()
        
    # initialize val_loss_accum that will be used to accumulate loss values
    # across an entire epoch for each batch and to compute the epoch loss
    
    val_loss_accum = 0
    
    # initialize val_num_corr_pred that will be used to accumulate the number
    # of correct predictions across an entire epoch for batch and to compute
    # the epoch validation accuracy
    
    val_num_corr_pred = 0    
    
    # compute the total number of samples in the validation set
    
    val_set_size = len(val_dataloader.dataset)
    
    # get the batch size for the validation set
    
    val_batch_size = val_dataloader.batch_size
    
    # iterate over all batches in validation set 
    
    for (images, weather_data, labels) in val_dataloader:
        
        # put batch of images onto GPU if available
        
        images = images.to(device)
        
        # put weather data onto GPU if available
        
        weather_data = weather_data.to(device).type(dtype)
        
        # normalize to avoid exploding and vanishing gradients
        
        weather_data = nn.functional.normalize(weather_data,p=2,dim=0)
        
        # put batch of labels onto GPU if available
        
        labels = labels.to(device)
        
        # NOTE: gradient tracking is NOT enabled since images and labels both have
        # requires_grad = False
        
        # zero parameter gradients
        
        optimizer.zero_grad()
        
        # no need to track gradients during inference phase
        
        with torch.set_grad_enabled(False):
            
            # forward propagation. No need to put outputs onto GPU because
            # images are already on GPU. This means tha outputs will automatically
            # be on GPU

            outputs = model(images, weather_data)
        
            # compute class predictions
        
            preds = torch.argmax(outputs, dim = 1)
        
            # compute loss. No need to put loss onto GPU since both outputs and
            # labels are on GPU. This means that loss will automatically be put
            # onto GPU. In validation phase, return loss as scalar
        
            loss = loss_func(outputs,labels).item()
        
        # accumulate loss for each batch
        
        val_loss_accum = val_loss_accum + (loss*val_batch_size)
        
        # accumulate number of correct predictions for each batch
        
        val_num_corr_pred = val_num_corr_pred + torch.sum(preds == labels)
                
    # update learning rate
    
    scheduler.step()
    
    # compute the average validation loss across the epoch
    
    epoch_val_loss = val_loss_accum / val_set_size
    
    # compute the validation accuracy for the epoch
    
    epoch_val_acc = val_num_corr_pred.item() / val_set_size
    
    return (model,
            epoch_train_acc,
            epoch_train_loss,
            epoch_val_acc,
            epoch_val_loss)

def test_model(model,test_data_loader,device):
    
    # put model in inference mode
    
    model.eval()
    
    class_predictions = []
    
    dtype = (torch.cuda.FloatTensor if torch.cuda.is_available()
             else torch.FloatTensor)
    
    for image,weather_data,label in test_data_loader:
        
        # put batch of images onto GPU if available
        
        image = image.to(device)
        
        # put weather data onto GPU if available
        
        weather_data = weather_data.to(device).type(dtype)
        
        # normalize to avoid exploding and vanishing gradients
        
        weather_data = nn.functional.normalize(weather_data,p=2,dim=0)
        
        output = model(image,weather_data)
        
        class_num = torch.argmax(output).item()
        
        class_predictions.append(class_num)
    
    histogram = [(sum([x == y for x in class_predictions])) for y in [0,1,2,3,4]]
    
    return (class_predictions,histogram)

def save_model(model,num_epochs,optimizer):
    
    save_settings = {
        
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        
    }

    torch.save(save_settings,'alexnet_'+str(num_epochs)+'.pt')
    
    return

def load_model(filename,model,optimizer):
    
    # load previously trained weights
    
    checkpoint = torch.load(filename)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    
    model.train()
    
    return model

def get_pretrained_model():
    
    class AlexNet(nn.Module):
        def __init__(self):
            super(AlexNet, self).__init__()
            self.alexnet = models.alexnet(pretrained = True,
                                          progress = True)
            
            # freeze convolutional layers and only train dense layers
    
            for param in self.alexnet.features.parameters():
                param.requires_grad = False
            
            self.alexnet.classifier[1] = nn.Linear(
                in_features = self.alexnet.classifier[1].in_features + 12,
                out_features = self.alexnet.classifier[1].out_features,
                bias = True)
            
            self.alexnet.classifier[6] = nn.Linear(in_features = 4096,
                                                   out_features = 5,
                                                   bias = True)
                        
        def forward(self, image, weather_data):
            
            x = self.alexnet.features(image)
            x = self.alexnet.avgpool(x)
            x1 = torch.flatten(x, 1)
            x = torch.cat((x1, weather_data), dim=1)
            x = self.alexnet.classifier(x)
            return x
        
    alexnet = AlexNet()

    return alexnet

def main():

    # initialize path to GPU if available
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # initialize pretrained model
    
    alexnet = get_pretrained_model().to(device)
    
    # final weights
    
    best_model_weights = copy.deepcopy(alexnet.state_dict())
    
    # final validation accuracy
    
    best_val_acc = 0
    
    # weighted loss function
    
    class_weights = torch.tensor([1,338/149,338/30,338/36,338/67]).to(device)
    
    loss_func = nn.CrossEntropyLoss(weight = class_weights)
    
    # optimizer

    optimizer = optim.SGD(alexnet.parameters(),
                          lr = 0.005,
                          momentum = 0.6)

    # learning rate scheduler

    scheduler = optim.lr_scheduler.StepLR(optimizer = optimizer,
                                          step_size = 5,
                                          gamma = 0.5,
                                          last_epoch = -1)
    
    # get the dataloaders for the training, validation, and testing sets
    
    input_img_size = 227
    
    img_dir = 'TrainData-C2/backup_copy_of_images_unsorted'
    
    train_csv = 'TrainData-C2/labels/ExtraCredit_Train.csv'
    
    train_dataloader = get_dataloader(input_img_size,
                                      train_csv,
                                      ImageAndWeatherDataset,
                                      img_dir,
                                      16)
    
    val_csv = 'TrainData-C2/labels/ExtraCredit_Test.csv'
    
    val_dataloader = get_dataloader(input_img_size,
                                    val_csv,
                                    ImageAndWeatherDataset,
                                    img_dir,
                                    16)
    
    test_dataloader = get_dataloader(input_img_size,
                                     val_csv,
                                     ImageAndWeatherDataset,
                                     img_dir,
                                     1)
    
    # number of epochs to train and vaildate for
    
    num_epochs = 20
    
    # used to track how long it takes to finish training and validation
    
    train_and_val_start = time.time()
    
    # save class predictions every 5 epochs
    
    preds_5epochs = []
    
    for epoch in range(num_epochs):
        
        epoch_start = time.time()
        
        # train and validate model every epoch
        
        (alexnet,
         train_acc,
         train_loss,
         val_acc,
         val_loss) = train_and_val_model(alexnet,
                                         optimizer,
                                         scheduler,
                                         loss_func,
                                         train_dataloader,
                                         val_dataloader,
                                         device)
        
        # show results
                                         
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 30)
        print('Training Loss: {:.4f}'.format(train_loss))
        print('Training Accuracy: {:.2f}%'.format(train_acc*100))
        print('Validation Loss: {:.4f}'.format(val_loss))
        print('Validation Accuracy: {:.2f}%'.format(val_acc*100))
        
        # save the weights for the best validation accuracy
        
        if val_acc > best_val_acc:
            
            best_val_acc = val_acc
            
            best_model_weights = copy.deepcopy(alexnet.state_dict())
        
        if ((epoch+1) % 2) == 0: # every 2 epochs
            
            # get predictions on the unlabelled test set
        
            class_predictions, histogram = test_model(alexnet,test_dataloader,device)
            
            # store these predictions
            
            preds_5epochs.append(class_predictions)
            
            # show the class histogram
            
            print('Class Histogram: {}'.format(histogram))
            
            # save the model
            
            save_model(alexnet,epoch+1,optimizer)
        
        epoch_end = time.time()
        
        epoch_time = time.strftime("%H:%M:%S",time.gmtime(epoch_end - epoch_start))
        
        print('Epoch Elapsed Time (HH:MM:SS): ' + epoch_time)
    
    train_and_val_end = time.time()
    
    train_and_val_time = time.strftime("%H:%M:%S",time.gmtime(train_and_val_end - 
                                                              train_and_val_start))
    
    print('\nTotal Training and Validation Time: ' + train_and_val_time)
    print('Best Validation Accuracy: {:.2f}%'.format(best_val_acc*100))
    
    return
        
if __name__ == '__main__':

    main()
