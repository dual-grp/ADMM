import torch
import os

from FLAlgorithms.users.userADMM import UserADMM
from FLAlgorithms.users.userADMM2 import UserADMM2
from FLAlgorithms.servers.serverbase2 import Server2
from utils.model_utils import read_data, read_user_data
import numpy as np

# Implementation for FedAvg Server

class ADMM(Server2):
    def __init__(self, experiment, device, dataset, learning_rate, ro, num_glob_iters, local_epochs, num_users, dim, time):
        super().__init__(device, dataset, learning_rate, ro, num_glob_iters, local_epochs, num_users, dim, time)

        # Initialize data for all  users
        self.K = 0
        self.experiment = experiment
        #oriDim = getDimention(0,dataset[0], dataset[1])
        total_users = len(dataset[0][0])
        for i in range(total_users):            
            id, train , test = read_user_data(i, dataset[0], dataset[1])
            #train = train# - torch.mean(train, 1)).T
            #test = test #- torch.mean(test, 1)).T
            if(i == 0):
                U, S, V = torch.svd(train)
                V = V[:, :dim]
                self.commonPCAz = V
                check = torch.matmul(V.T,V)

            #user = UserADMM(device, id, train, test, self.commonPCAz, learning_rate, ro, local_epochs, dim)
            user = UserADMM2(device, id, train, test, self.commonPCAz, learning_rate, ro, local_epochs, dim)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")

    def train(self):
        for glob_iter in range(self.num_glob_iters):
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: ",glob_iter, " -------------")
            #loss_ = 0
            self.send_pca()

            # Evaluate model each interation
            self.evaluate()

            self.selected_users = self.select_users(glob_iter,self.num_users)
            
            #NOTE: this is required for the ``fork`` method to work
            for user in self.users[0:3]:
                user.train(self.local_epochs)

            self.aggregate_pca()
            
        self.save_results()
        self.save_model()