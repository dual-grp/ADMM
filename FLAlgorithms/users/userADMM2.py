import torch
import os
import json
from FLAlgorithms.users.userbase import User
import copy
# Implementation for FedAvg clients

class UserADMM2():
    def __init__(self, device, id, train_data, test_data, commonPCA, learning_rate, ro, local_epochs, dim):
        self.localPCA   = copy.deepcopy(commonPCA) # local U
        self.localZ     = copy.deepcopy(commonPCA)
        self.localY     = copy.deepcopy(commonPCA)
        self.localT     = torch.matmul(self.localPCA.T, self.localPCA)
        self.ro = ro
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
        # update local Y
        self.localZ = commonPCA.data.clone()
        self.localY = self.localY + self.ro * (self.localPCA - self.localZ)
        # update local T
        temp = torch.matmul(self.localPCA.T, self.localPCA) - torch.eye(self.localPCA.shape[1])
        hU = torch.max(torch.zeros(temp.shape),temp)**2
        self.localT = self.localT + self.ro * hU

    def train_error_and_loss(self):
        residual = torch.matmul((torch.eye(self.localPCA.shape[0]) - torch.matmul(self.localPCA, self.localPCA.T)), self.train_data)
        loss_train = torch.norm(residual, p="fro") ** 2 / self.train_samples
        return loss_train , self.train_samples

    def hMax(self):
        temp = torch.matmul(self.localPCA.T, self.localPCA) - torch.eye(self.localPCA.shape[1])
        return torch.max(torch.zeros(temp.shape),temp)#torch.max(0,torch.eye(U[1])- torch.matmul(U.T, U))

    def train(self, epochs):
        print("Client--------------",self.id)
        for i in range(self.local_epochs):
            self.localPCA.requires_grad_(True)
            residual = torch.matmul(torch.eye(self.localPCA.shape[0])- torch.matmul(self.localPCA, self.localPCA.T), self.train_data)
            temp = torch.matmul(self.localPCA.T, self.localPCA) - torch.eye(self.localPCA.shape[1])
            hU = torch.max(torch.zeros(temp.shape),temp)**2
            regularization = 0.5 * self.ro * torch.norm(self.localPCA - self.localZ)** 2 + 0.5 * self.ro * torch.norm(hU) ** 2
            frobenius_inner = torch.sum(torch.inner(self.localY, self.localPCA - self.localZ)) + torch.sum(torch.inner(self.localT, hU))
            self.loss = 1/self.train_samples * torch.norm(residual, p="fro") ** 2 
            #print("self.loss", self.loss.data)
            self.lossADMM = self.loss + 1/self.train_samples * (frobenius_inner + regularization)
            #print("self.loss", self.loss)
            #print("self.lossADMM", self.lossADMM)
            temp = self.localPCA.data.clone()
            # slove local problem locally
            if self.localPCA.grad is not None:
                self.localPCA.grad.data.zero_()

            self.lossADMM.backward(retain_graph=True)
            #localGrad = self.localPCA.grad.data.clone()# grad[0]
            # update local pca
            temp  = temp - self.learning_rate * self.localPCA.grad
            self.localPCA = temp.data.clone()
        return 1