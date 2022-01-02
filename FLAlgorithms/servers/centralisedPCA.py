import torch


from FLAlgorithms.users.userADMMPCA import UserADMMPCA
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.trainmodel.models import *
from utils.model_utils import read_data, read_user_data

import numpy as np

class Centralised(Server):
    def __init__(self, experiment, device, dataset, learning_rate, ro, num_glob_iters, local_epochs, num_users, dim, time):
        super().__init__(device, dataset, learning_rate, ro, num_glob_iters, local_epochs, num_users, dim, time)

        # Initialize data for all  users
        self.K = 0
        self.experiment = experiment
        
        total_users = len(dataset[0][0])
        print("total users: ", total_users)
        for i in range(total_users):            
            id, train , test, y_train, y_test = read_user_data(i, dataset[0], dataset[1])
          
            if(i == 0):
                U, S, V = torch.svd(train)
                V = V[:, :dim]
                self.commonPCAz = V
                print("type of V", type(V))
                print("shape of V: ", V.shape)
                print(self.commonPCAz)
                # check = torch.matmul(V.T,V)
            
            user = UserADMMPCA(device, id, train, test, y_train, y_test, self.commonPCAz, learning_rate, ro, local_epochs, dim)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")

    def train(self):
        losses = []
        loss = self.evaluate()
        losses.append(loss.item())

        self.send_pca()
        print("Centralised PCA Completed training!!!")
        print(losses)
        
        return losses
    
    def test(self, model, dataset, global_epoch):
        # Get device status: Check GPU or CPU
        device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
        
        if(dataset == "Mnist" or dataset == "Cifar10"):
            model = DNN2(self.dim, 100, 100,10).to(device), model
        else:
            model = DNN2(self.dim, 20, 20, 10).to(device), model
        accuracy = 0
        batch_size = 2
        for user in self.users: 
           accuracy += user.test(model, dataset, global_epoch, batch_size)
        accuracy = accuracy / len(self.users)
        print("------ averaged accuracy ",accuracy,"---------")
        return accuracy
                
    

            

        