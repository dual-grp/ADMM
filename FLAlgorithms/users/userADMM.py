import torch
import os
import json
from FLAlgorithms.users.userbase import User
import copy
# Implementation for FedAvg clients

class UserADMM():
    def __init__(self, device, id, train_data, test_data, commonPCA, learning_rate, ro, local_epochs, dim):
        self.localPCA = copy.deepcopy(commonPCA)
        self.ro = ro
        self.localZ = copy.deepcopy(commonPCA)
        self.localY = copy.deepcopy(commonPCA)
        self.device = device
        self.id = id
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.dim = dim
        self.train_data = train_data.T
        self.test_data = test_data.T
        self.localPCA.requires_grad_(True)

    def set_commonPCA(self, commonPCA):
        self.localZ = commonPCA.data.clone()
        self.localY = self.localY + self.ro * (self.localPCA - self.localZ)

    def train_error_and_loss(self):
        residual = torch.matmul((torch.eye(self.localPCA.shape[0]) - torch.matmul(self.localPCA, self.localPCA.T)), self.train_data)
        loss_train = torch.norm(residual, p="fro") ** 2 / self.train_samples
        return loss_train , self.train_samples

    def train(self, epochs):
        print("Client--------------",self.id)
        for i in range(self.local_epochs):
            self.localPCA.requires_grad_(True)
            residual = torch.matmul(torch.eye(self.localPCA.shape[0])- torch.matmul(self.localPCA, self.localPCA.T), self.train_data)
            #print(residual.shape)
            regularization = 0.5 * self.ro * torch.norm(self.localPCA - self.localZ + 1/self.ro * self.localY) ** 2
            #print(regularization)
            # get Euclidean gradient
            self.loss = 1/self.train_samples * torch.norm(residual, p="fro") ** 2 
            
            self.lossADMM = self.loss + regularization
            #print("self.loss", self.loss)
            #print("self.lossADMM", self.lossADMM)
            self.lossADMM.backward(retain_graph=True)

            # project Euclidean gradient Fxi on the tangent space
            projected_grad = torch.matmul(torch.eye(self.localPCA.shape[0])- torch.matmul(self.localPCA, self.localPCA.T),self.localPCA.grad)
            
            # mapping tangent matrix back to the manifold
            temp = self.localPCA.data.clone()

            if self.localPCA.grad is not None:
                self.localPCA.grad.data.zero_()

            temp  = temp + self.learning_rate * projected_grad
            q, r = torch.linalg.qr(temp)
            self.localPCA = q.data.clone()
        return 1