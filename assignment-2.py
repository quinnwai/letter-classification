#!/usr/bin/python3

# data processing
import numpy as np
import pandas as pd

# preprocessing and model packages
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

# neural net packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

# timing and graph settings
from time import time
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

'''Common Functions: used in all methods'''
def plotter(x, y, title,xlabel,path):
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Validation Error")
    plt.savefig(path)

'''5 Model Functions'''
def kNeighbors(X, y, X_test, y_test, k_vals, title=""):
    # log start time
    t0 = time()
    val_errors, test_errors = np.zeros(len(k_vals)), np.zeros(len(k_vals))


    # calculate validation error for each model
    for i, k in enumerate(k_vals):
        # define kfolds
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        

        # 5-fold validation of kNN for given k
        score = 0
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx,:], X[val_idx,:]
            y_train, y_val = y[train_idx], y[val_idx]

            model = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
            score += model.score(X_val, y_val)

        # store validation error
        val_errors[i] = 1 - score/n_splits

        # calculate test error
        final_model = KNeighborsClassifier(n_neighbors=k).fit(X,y)
        test_errors[i] = 1 - final_model.score(X_test,y_test)
    
    # determine error of model based on validation error
    best_model_idx = val_errors == val_errors.min()
    test_error = test_errors[best_model_idx][0]
    k = k_vals[best_model_idx][0]

    # # plot validation errors for different x values 
    # figtitle = f"{title}: Tuning k in k Nearest Neighbors"
    # xlabel = "k"
    # path = f"figures/{title}: knn tuning"
    # plotter(k_vals, val_errors, figtitle, xlabel, path)

    # record time
    t1 = time()
    runtime = t1 - t0

    return val_errors, k, test_error, runtime

def decisionTree(X, y, X_test, y_test, min_leaf_vals, title=""):
    # log start time
    t0 = time()
    val_errors, test_errors = np.zeros(len(min_leaf_vals)), np.zeros(len(min_leaf_vals))


    # get test error for each
    for i, min_samples_leaf in enumerate(min_leaf_vals):
        # define k folds
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        

        # 5-fold validation of decision tree given a minimum number of samples per leaf
        score = 0
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx,:], X[val_idx,:]
            y_train, y_val = y[train_idx], y[val_idx]

            model = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
            model.fit(X_train,y_train)
            score += model.score(X_val, y_val)

        # store validation error
        val_errors[i] = 1 - score/n_splits

        # calculate test error
        final_model = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
        final_model.fit(X,y)
        test_errors[i] = 1 - final_model.score(X_test,y_test)
    
    # determine error of model based on validation error
    best_model_idx = val_errors == val_errors.min()
    test_error = test_errors[best_model_idx][0]
    min_samples_leaf = min_leaf_vals[best_model_idx][0]

    # # plot scores 
    # figtitle = f"{title}: Tuning Minimum Samples per Leaf in a Decision Tree"
    # xlabel = "Minimum Samples per Leaf"
    # path = f"figures/{title}: decision tree tuning"
    # plotter(min_leaf_vals, val_errors, figtitle, xlabel, path)

    # record time
    t1 = time()
    runtime = t1 - t0 

    return val_errors, min_samples_leaf, test_error, runtime 

def randomForest(X, y, X_test, y_test, n_estimators_vals, title=""):
    # log start time
    t0 = time()
    val_errors, test_errors = np.zeros(len(n_estimators_vals)), np.zeros(len(n_estimators_vals))


    # get test error for each
    for i, n_estimators in enumerate(n_estimators_vals):
        # define k folds
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        
        # 5-fold validation of decision tree given a minimum number of samples per leaf
        score = 0
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx,:], X[val_idx,:]
            y_train, y_val = y[train_idx], y[val_idx]

            model = RandomForestClassifier(n_estimators=n_estimators, random_state=1)
            model.fit(X_train,y_train)
            score += model.score(X_val, y_val)

        # store validation error
        val_errors[i] = 1 - score/n_splits

        # calculate test error
        final_model = RandomForestClassifier(n_estimators=n_estimators)
        final_model.fit(X,y)
        test_errors[i] = 1 - final_model.score(X_test,y_test)
    
    # determine error of model based on validation error
    best_model_idx = val_errors == val_errors.min()
    test_error = test_errors[best_model_idx][0]
    n_estimators = n_estimators_vals[best_model_idx][0]

    # # plot scores 
    # figtitle = f"{title}: Tuning Number of Estimators in a Random Forest"
    # xlabel = "Number of Estimators"
    # path = f"figures/{title}: decision tree tuning"
    # plotter(n_estimators_vals, val_errors, figtitle, xlabel, path)

    # record time
    t1 = time()
    runtime = t1 - t0 

    return val_errors, n_estimators, test_error, runtime 

def supportVectorMachine(X, y, X_test, y_test, C_vals, title=""):
    # log start time
    t0 = time()
    val_errors, test_errors = np.zeros(len(C_vals)), np.zeros(len(C_vals))


    # get test error for each
    for i, C in enumerate(C_vals):
        # define k folds
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        
        # 5-fold validation of decision tree given a minimum number of samples per leaf
        score = 0
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx,:], X[val_idx,:]
            y_train, y_val = y[train_idx], y[val_idx]

            model = svm.SVC(C=C)
            model.fit(X_train,y_train)
            score += model.score(X_val, y_val)

        # store validation error
        val_errors[i] = 1 - score/n_splits

        # calculate test error
        final_model = svm.SVC(C=C)
        final_model.fit(X,y)
        test_errors[i] = 1 - final_model.score(X_test,y_test)
    
    # determine error of model based on validation error
    best_model_idx = val_errors == val_errors.min()
    test_error = test_errors[best_model_idx][0]
    C = C_vals[best_model_idx][0]

    # # plot scores 
    # figtitle = f"{title}: Tuning Regularization Constant (C) in SVM"
    # xlabel = "C"
    # path = f"figures/{title}: SVM tuning"
    # plotter(C_vals, val_errors, figtitle, xlabel, path)

    # record time
    t1 = time()
    runtime = t1 - t0 

    return val_errors, C, test_error, runtime 

def neuralNetwork(X, y, X_test, y_test, epoch_vals, title="", net_type=OriginalNet, learning_rate=1e-3):
    # log start time
    t0 = time()
    val_errors = np.zeros(len(epoch_vals))

    # define hyperparameters
    batch_size = 5
    epochs = 50
    loss_fn = nn.CrossEntropyLoss() # loss function for classification
    
    # ensure correct datatypes for X and y datasets 
    num_X, num_X_test = X.astype('int8'), X_test.astype('int8')
    labels = np.unique(y)
    num_y = (y == labels[0])
    num_y_test = (y_test == labels[0])

    # define dataloaders for test set
    tensor_X_test, tensor_y_test = torch.Tensor(num_X_test), torch.Tensor(num_y_test).type(torch.LongTensor)

    # create test set iterable using dataset
    data_test = TensorDataset(tensor_X_test,tensor_y_test)
    testloader = DataLoader(data_test, batch_size=batch_size)
    
    # select learning rate with the model with the lowest validation error
    for i, epochs in enumerate(epoch_vals):
    # for i, epochs in enumerate([epochs]):
        # define k folds
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        
        # 5-fold validation of decision tree given a minimum number of samples per leaf
        score = 0
        for train_idx, val_idx in kf.split(num_X):
            # initialize neural net
            net = net_type()
            optimizer = optim.SGD(net.parameters(), lr=learning_rate)

            # create train and validation datasets as tensors
            X_train, y_train = num_X[train_idx,:], num_y[train_idx]
            X_val, y_val = num_X[val_idx,:], num_y[val_idx]
            tensor_X_train, tensor_y_train = torch.Tensor(X_train), torch.Tensor(y_train).type(torch.LongTensor)
            tensor_X_val, tensor_y_val = torch.Tensor(X_val), torch.Tensor(y_val).type(torch.LongTensor)

            # create dataloaders (iterables for datasets)
            data_train = TensorDataset(tensor_X_train,tensor_y_train)
            data_val = TensorDataset(tensor_X_val,tensor_y_val)
            trainloader, valloader = DataLoader(data_train, batch_size=batch_size), DataLoader(data_val, batch_size=batch_size)

            # train for multiple epochs
            train_epochs(epochs, net, trainloader, optimizer, loss_fn)

            # calculate validation error and add to score
            correct = 0
            total = 0
            with torch.no_grad():
                for data in valloader:
                    images, labels = data
                    # calculate outputs by running images through the network
                    outputs = net(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            score += correct/total

        # store validation error
        val_errors[i] = 1 - score/n_splits
        print(f"val:  {val_errors[i]}")

    # determine best learning rate based on validation error
    best_model_idx = val_errors == val_errors.min()
    learning_rate = epoch_vals[best_model_idx][0]

    # use best learning rate to train model 
    net = net_type()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    tensor_X, tensor_y = torch.Tensor(num_X), torch.Tensor(num_y).type(torch.LongTensor)
    data_train = TensorDataset(tensor_X,tensor_y)
    trainloader = DataLoader(data_train, batch_size=batch_size)
    train_epochs(epochs, net, trainloader, optimizer, loss_fn)

    # calculate test error using best model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_error = 1 - score/n_splits

    # # plot scores 
    # figtitle = f"{title}: Tuning Number of Epochs in NN"
    # xlabel = "Number of Epochs"
    # path = f"figures/{title}: NN tuning"
    # plotter(epoch_vals, val_errors, figtitle, xlabel, path)

    # record time
    t1 = time()
    runtime = t1 - t0 

    return val_errors, learning_rate, test_error, runtime 

'''Neural Network Classes/Helper Methods'''
class OriginalNet(nn.Module):
    def __init__(self):
        super(OriginalNet, self).__init__()

        # # dropout layer for regularization / prevent overfitting
        # self.dropout1 = nn.Dropout2d(0.25)

        # two fully connected layers
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 2)

    # x represents our data
    def forward(self, x):
        # Pass data through fc1, then apply relu
        x = self.fc1(x)
        x = F.relu(x)

        # dropout layer before passing data to fc2
        # x = self.dropout1(x)
        x = self.fc2(x)

        # Apply softmax to x
        output = F.log_softmax(x, dim=1)
        return output

class ReducedNet(nn.Module):
    def __init__(self):
        super(ReducedNet, self).__init__()

        # # dropout layer for regularization / prevent overfitting
        # self.dropout1 = nn.Dropout2d(0.25)

        # two fully connected layers
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    # x represents our data
    def forward(self, x):
        # Pass data through fc1, then apply relu
        x = self.fc1(x)
        x = F.relu(x)

        # dropout layer before passing data to fc2
        # x = self.dropout1(x)
        x = self.fc2(x)

        # Apply softmax to x
        output = F.log_softmax(x, dim=1)
        return output

def train_epochs(epochs, net, trainloader, optimizer, loss_fn):
    for epoch in range(epochs):
        running_loss = 0.0
        for j, data in enumerate(trainloader):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if j % 100 == 99: running_loss = 0.0

def main():
    # # Data preprocessing
    # load in data 
    data = pd.read_csv("data/letter-recognition.data", delimiter=",",header=None).to_numpy()

    # get indices corresponding to each classification
    hk_idxs = np.logical_or(data[:,0] == "H", data[:,0] == "K")
    my_idxs = np.logical_or(data[:,0] == "M", data[:,0] == "Y")
    do_idxs = np.logical_or(data[:,0] == "D", data[:,0] == "O")
    data_idxs = [hk_idxs, my_idxs, do_idxs]
    data_titles = ["H vs K", "M vs Y", "D vs O"]
    fig_markers = ["o", "v", "s"]

    # separate into train vs test and features (X) vs label (Y) data for each classification problem
    X_train_sets, y_train_sets = [0]*len(data_idxs), [0]*len(data_idxs)
    X_test_sets, y_test_sets = [0]*len(data_idxs), [0]*len(data_idxs)

    for i, idxs in enumerate(data_idxs):
        data_train, data_test = train_test_split(data[idxs,:], test_size=0.1, random_state=420)
        X_train_sets[i], y_train_sets[i] = data_train[:,1:],  data_train[:,0]
        X_test_sets[i], y_test_sets[i] = data_test[:,1:], data_test[:,0]
    
    # list of model functions
    models = [kNeighbors, decisionTree, randomForest, supportVectorMachine, neuralNetwork]

    # create list from lists of hyperparamters values
    k_vals = np.arange(1,11,2)
    min_leaf_vals = np.array([1,5,10,15,20])
    n_estimators_vals = np.array([50, 100, 150, 200, 250])
    C_vals = np.array([1.0, 2.0, 2.5, 3.0, 3.5])
    epoch_vals = np.array([10, 30, 50, 70])
    hyperparams = [k_vals, min_leaf_vals, n_estimators_vals, C_vals, epoch_vals]

    # model-specific strings for plots / filenames
    models_str = ["kNN", "Decision Tree", "Random Forest", "SVM", "Neural Network"]
    hyperparams_str = ["k", "min leaf", "num bags", "C", "epochs"]
    xlabels = ["k", "Minimum Samples per Leaf", "Number of Estimators", "C", "Number of Epochs"]

    ## Dimension Reduction (using only training data)
    print("~Dimension Reduction~")
    # store features used for each dataset
    ft_idxs = [np.ones(X_train_sets[0].shape[1], dtype=bool) for i in data_idxs]

    # 1: Drop 2 feature with the lowest variance
    for k, idxs in enumerate(data_idxs):
        X_train = X_train_sets[k].astype(float)
        stds = X_train.std(axis=0)
        min_std = np.sort(stds)[1]
        ft_idxs[k] = np.logical_and(ft_idxs[k], stds > min_std)
        idxs = list(np.where(stds <= min_std))
        print(f"{data_titles[k]}: dropping 2 features with std ≤ {min_std}: {idxs}")

    # 2: Drop 1 feature with high correlations
    print("\nList of correlations above 0.8")
    for k, idxs in enumerate(data_idxs):
        X_train = X_train_sets[k]
        for i in range(X_train.shape[1]):
            for j in range(i+1, X_train.shape[1]):
                (cor, p) = pearsonr(X_train[:,i], X_train[:,j])
                if cor > 0.8:
                    print(f"cor ({i},{j}): {cor, p}")
                    
    
    print("dropping feature 0 from all datasets...\n")
    for ft_idx in ft_idxs:
        ft_idx[0] = False

    # 3: Using remaining features, get 4 primary components from PCA
    # normalize datasets
    X_train_norm_sets = [StandardScaler().fit_transform(X_train) for X_train in X_train_sets]
    X_test_norm_sets =  [StandardScaler().fit_transform(X_test) for X_test in X_test_sets]

    # run pca
    pcas, X_train_pca_sets, X_test_pca_sets = [0] * len(data_idxs), [0] * len(data_idxs), [0] * len(data_idxs)
    for i, X in enumerate(X_train_norm_sets):
        X_train = X[:,ft_idx]
        X_test = X_test_norm_sets[i][:,ft_idx]
        pcas[i] = PCA(n_components=4)
        pca = pcas[i]
        X_train_pca_sets[i] = pca.fit_transform(X_train)
        X_test_pca_sets[i] = pca.transform(X_test)
        print(f"{data_titles[i]}: {pca.explained_variance_ratio_}, {pca.explained_variance_ratio_.sum()}")
    print("")


    # # Model Fitting
    # for normal and reduced dataset
    NORMAL, REDUCED = range(2)
    type_str = ["Normal", "Reduced"]
    for type in range(2):
        # title
        if type == NORMAL: print("~Normal Datasets~")
        else: print("~Reduced Datasets~")
        

        # for models given above (all but ANN), select the best 
        for i, model in enumerate(models):
            # initialize model plots
            plt.figure()
            plt.title(f"{type_str[type]} Dataset: Tuning {xlabels[i]} in {models_str[i]}")
            plt.xlabel(xlabels[i])
            plt.ylabel("Validation Error")

            # for each datasets train model
            print(f"{models_str[i]} \n{hyperparams_str[i]} \t\ttest error \truntime")
            for j, idxs in enumerate(data_idxs):
                # create train and test data
                if type == NORMAL:
                    X_train, X_test = X_train_sets[j], X_test_sets[j]  
                elif type == REDUCED:
                    X_train, X_test = X_train_pca_sets[j], X_test_pca_sets[j]
                
                y_train, y_test = y_train_sets[j], y_test_sets[j]

                # run model in question to get optimal hyperparameter values, error, and runtime
                if i == 4 and type == REDUCED:
                    val_errors, hyperparam, test_error, runtime = model(X_train, y_train, X_test, y_test, hyperparams[i],
                                title=f"{type_str[type]} {data_titles[j]}", net_type=ReducedNet, learning_rate=5e-2)
                else:
                    val_errors, hyperparam, test_error, runtime = model(X_train, y_train, X_test, y_test, hyperparams[i],
                                title=f"{type_str[type]} {data_titles[j]}")
                print(f"{hyperparam} \t\t{test_error:.4f} \t\t{runtime:.4f}")

                # add plot of validation error
                plt.plot(hyperparams[i], val_errors, label=data_titles[j], marker=fig_markers[j])
            
            # save final plot
            plt.legend()
            plt.savefig(f"figures/{type_str[type].lower()}_tuning_{models_str[i].replace(' ', '_')}")

            print("")

        print("\n------------------\n\n")


if __name__ == "__main__":
    main()

