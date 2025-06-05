import torch
import os
import argparse
import pandas as pd
from torch import nn
import numpy as np #used for matrix operations
from torch.utils.data import DataLoader, Dataset, ConcatDataset #concatenation of the train_data and the test_data
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, v2
from torchvision.io import read_image 
from timeit import default_timer as timer   
from sklearn.model_selection import train_test_split  
            
#conda (de)activate lumiere

#Setting up for the Classification Model
class research(Dataset):
    def __init__(self, csv_dir, image_dir, transform = None, target_transform = None):
        self.labels = pd.read_csv(csv_dir) 
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform # if u want to modify the labels
        
    def __len__(self):
        #return len(self.labels)
        return len(self.labels)
    
    def __getitem__(self, index):   
        imagePath = self.labels.iloc[index, 0] #relative file path
        curImage = read_image(imagePath)
        curLabel = self.labels.iloc[index, 1]
        #curTensor = torch.from_numpy(curImage) # make a tensor from an image
        
        curImage = self.transform(curImage)
        if self.target_transform: 
            curLabel = self.target_transform(curLabel)
        return curImage, curLabel


def getLoaders(batchSize):
    
    current_dir = os.getcwd() # Get the current working directory
    image_dir_name = "train_data_beginning" # Specify the name of the directory containing the images
    image_dir_name_all_test_data = "the_actual_test_data"
    csv_dir_name = "researchProjectMapLabels.csv"
    csv_dir_name_all_test_data = "the_actual_test_data.csv"
    real_data_name = "real_data" 
    csv_real_data_name = "real_data_dataset.csv" # the only directory that has real data 
    
    image_dir = os.path.join(current_dir, image_dir_name) # Construct the comprehensive path to the directory ; for the train dataloader 
    csv_dir = os.path.join(current_dir, csv_dir_name) 
    csv_dir_test = os.path.join(current_dir, csv_dir_name_all_test_data) 
    csv_real_data = os.path.join(current_dir, csv_real_data_name) 
    device = "cuda" 

    test_transform = transforms.Compose([ # prepare the batch of images before (doing operations on them and) making a dataloader for the dataset
    transforms.Resize((224,224)), 
    v2.RandomResizedCrop(size = (224, 224), antialias = True), # data augmentation 
    v2.RandomHorizontalFlip(p =  0.5)
    # transforms.Normalize(mean, std) # simplifies rgb values to [0.0, 1.0] instead of [0, 255] using the mean and the mean standard deviation to enhance performance and optimize
    # transforms.ToTensor() #for some reason, VS Code automatically converts the transforms into tensors 
    ]) 

    train_data = research(csv_dir, image_dir, transform = test_transform, target_transform = None)
    test_data = research(csv_dir_test, image_dir_name_all_test_data, transform = test_transform, target_transform = None) 
    
    all_data = ConcatDataset([train_data, test_data]) 
    
    all_real_data = research(csv_real_data, real_data_name, transform = test_transform, target_transform = None) 
    
    train_and_val, test = train_test_split(all_data, test_size = 0.15, train_size = 0.85, random_state = 0) 
    train, val = train_test_split(train_and_val, test_size = 0.17, train_size = 0.83, random_state = 0) 

    # test = all_real_data # put all of the real data in the test dataset
    
    trainLoader = DataLoader(train, batch_size = batchSize, shuffle = True) 
    testLoader = DataLoader(test, batch_size = batchSize, shuffle = True)
    valLoader = DataLoader(val, batch_size = batchSize, shuffle = True)
    
    return trainLoader, testLoader, valLoader, train, val, test



# https://www.youtube.com/watch?v=n8Mey4o8gLc
# https://www.youtube.com/watch?v=V_xro1bcAuA




#      input_shape = # of color channels (rgb)

#3 conv and relu instead of 2, 1 conv and relu instead of 2, number of layers, instead of relu look at different activation functions (some might not work), instead of max pool do average pool, try different stride amounts up to kernel size or 3
# add another evaluation/test section where you only test on real data. this means you have 2 evaluation/test sections (pick 5 main learning rates (unless im adventurous) and do the same graph as "learning rate vs accuracy (general)") 

class NeuralNetwork(nn.Module): #nn.Module is the basis for all neural networks 
    def __init__(self, input_shape, hidden_units, output_shape, kernels, adding, pooling): #sets up the network's structure by defining the layers 
        super().__init__() 
        self.layer1 = nn.Sequential( 
            nn.Conv2d(in_channels = input_shape, out_channels = hidden_units, kernel_size = kernels, stride = 1, padding = 1), 
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units + adding, kernel_size = kernels, stride = 1, padding = 1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = pooling) 
        ) 
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units + adding, out_channels = hidden_units + adding + adding, kernel_size = kernels, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units + adding + adding, out_channels = hidden_units + adding, kernel_size = kernels, stride = 1, padding = 1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = pooling) 
        ) 
        ''' 
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size = kernels, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size = kernels, stride = 1, padding = 1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = pooling)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size = kernels, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size = kernels, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = pooling)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size = kernels, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size = kernels, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = pooling)
        )
        '''
        x = torch.randn(1, 3, 224, 224)
        Out = self.layer1(x) 
        Out = self.layer2(Out) 
        # Out = self.layer3(Out) 
        # Out = self.layer4(Out) 
        # Out = self.layer5(Out) 
        
        # print(Out.shape)
        # print(np.prod(list(Out.shape))) 
        self.classifier = nn.Sequential( 
            nn.Flatten(), #replace the '0' 
            nn.Linear(in_features = np.prod(list(Out.shape)), 
                      out_features = output_shape)
        )
    def forward(self, image): #specifies how the data flows through the network
        #images = the input tensor to the model. it can be any data that the model is supposed to process
        image = self.layer1(image) 
        image = self.layer2(image) 
        # image = self.layer3(image) 
        # image = self.layer4(image) 
        # image = self.layer5(image) 
        
        image = self.classifier(image) 
        return image 

   

def run(rate, input_shape, hidden_units, output_shape, kernels, adding, pooling):
    # Hyperparameters
    numEpochs = 40 
    batchSize = 20 
    learning_rate = rate   
    
    trainLoader, testLoader, valLoader, train, val, test = getLoaders(batchSize) 
    model = NeuralNetwork(input_shape, hidden_units, output_shape, kernels, adding, pooling) 

    # print(len(val)) 44
    # print(len(test)) 45
    
    
    diction = {
        "happy": 0, 
        "sad": 1, 
        "angry": 2
    } 
    

    lossFunction = nn.CrossEntropyLoss() # Loss Function
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) # Optimizer Function

    # print(f"# of paramters: {sum(p.numel() for p in model.parameters())}")

    #Measure Start Time

    #Train and Test Model  
    maxValid, finalValid, finalTrain, finalTest = 0, 0, 0, 0  
    for epoch in range(1, numEpochs+1): 
        
        print(f"Epoch: {epoch}")
        train_samples, train_correct, val_samples, val_correct = 0, 0, 0, 0
        
        for x, y in trainLoader: # x = batch[0] , y = batch[1] training and then testing on train
            model.train() 
            numbers = []
            for i in y:
                numbers.append(diction[i])
            numbers = torch.Tensor(numbers).long() # why do we need .long() again?
            
            out = model(x.float())
            loss = lossFunction(out, numbers)
            #print(loss.item()) 
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() #reset (to 0s) 
            
            model.eval()
            # for x,y in trainLoader: # x = batch[0] , y = batch[1]
            
            # Forward Pass 
            _, predicted = torch.max(out, dim = 1) # explain
            
            # Update the running total of correct predictions and samples
            train_correct += (predicted == numbers).sum().item() # explain
            train_samples += numbers.size(0)  # explain
            
            #print(batch[0].dtype)
            #print(batch[0][0].dtype)
        finalTrain = 100 * train_correct / train_samples 
        
        # ---------------------------------------------------
        for x, y in valLoader: #testing on validation
            
            numbers = []
            for i in y:
                numbers.append(diction[i])
            numbers = torch.Tensor(numbers).long() # why do we need .long() again?
            
            # Forward Pass 
            out = model(x.float())
            _, predicted = torch.max(out, dim = 1) # explain
            
            # Update the running total of correct predictions and samples
            val_correct += (predicted == numbers).sum().item() # explain
            val_samples += numbers.size(0)  # explain

        accuracy = 100 * val_correct / val_samples
        maxValid = max(maxValid, accuracy)
        finalValid = accuracy
        
        # ---------------------------------------------------
        model.train()
        for x, y in valLoader: #training on validation
            numbers = []
            for i in y:
                numbers.append(diction[i])
            numbers = torch.Tensor(numbers).long()
            out = model(x.float())
            loss = lossFunction(out, numbers) 
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() #reset (to 0s) 
        
        # ---------------------------------------------------
        model.eval()
        correctTest, sizeTest = 0, 0 
        for x, y in testLoader: # testing on test
            numbers = []
            for i in y: 
                numbers.append(diction[i])
            numbers = torch.Tensor(numbers).long() 
            
            out = model(x.float())
            _, predTest = torch.max(out, dim = 1) 
            correctTest += (predTest == numbers).sum().item()
            sizeTest += numbers.size(0)
            
        finalTest = 100 * correctTest / sizeTest
        # ---------------------------------------------------
        
        print(f"valid accuracy: {finalValid}%")  
        print(f"test accuracy: {finalTest}%")  
        print()     
        
        
    print(f"Final Test: {finalTest}")
    print(f"Final Valid: {finalValid}") 
    print(f"Final Train: {finalTrain}") 
    print(f"Maximum Valid: {maxValid}") 
        
if __name__ == '__main__': 
    
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type = float)
    parser.add_argument('--kernels', type = int) 
    parser.add_argument('--hidden', type = int) 
    parser.add_argument('--adding', type = int)
    args = parser.parse_args() 
    lr = args.lr 
    kernels = args.kernels 
    hidden_units = args.hidden 
    adding = args.adding 
    '''

    #trainLoader, testLoader, valLoader, train, val, test = getLoaders(20)
    train_time_start_model = timer() 
    lr, input_shape, hidden_units, output_shape, kernels, adding, pooling = 4e-3, 3, 10, 3, 3, 0, 2   
    
    
    
    input_shape, output_shape, pooling = 3, 3, 2     # kernels = 3, 5, 7, etc. 
    run(lr, input_shape, hidden_units, output_shape, kernels, adding, pooling) 
    
    #print(len(trainLoader))
    #print(len(testLoader))
    #print(len(valLoader))
    train_time_end_model = timer() 
    print(f"{train_time_end_model - train_time_start_model} seconds") 
    print(f"Learning Rate: {lr}") 
    print(f"Kernel Size: {kernels}")
    print(f"Hidden Units: {hidden_units}") 
    print(f"Adding: {adding}")  
    print(f"Pooling: {pooling}") 
    
    print()
