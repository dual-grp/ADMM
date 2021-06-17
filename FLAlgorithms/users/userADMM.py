import torch
import os
import json
from FLAlgorithms.users.userbase import User
import copy
# Implementation for FedAvg clients

class UserADMM():
    def __init__(self, device, numeric_id, train_data, test_data, commonPCA, learning_rate, ro, local_epochs, dim):
        self.localPCA = copy.deepcopy(commonPCA)
        self.ro = ro
        self.localZ = copy.deepcopy(commonPCA)
        self.localLamda = copy.deepcopy(commonPCA)
        self.device = device
        self.id = id
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.dim = dim
        self.train_data = train_data
        self.test_data = test_data
        self.localPCA.requires_grad_(True)

    def set_commonPCA(self, commonPCA):
        self.localZ = commonPCA
        self.localLamda = self.localLamda + self.ro * (self.localPCA - self.localZ)

    def train_error_and_loss(self):
        residual = torch.matmul((torch.eye(self.localPCA.shape[0]) - torch.matmul(self.localPCA, self.localPCA.T)), self.train_data)
        loss_train = torch.norm(residual, p="fro") ** 2
        return loss_train , self.train_samples

    def train(self, epochs):
        for i in range(self.local_epochs):
            self.localPCA.requires_grad_(True)
            residual = torch.matmul((torch.eye(self.localPCA.shape[0]) - torch.matmul(self.localPCA, self.localPCA.T)), self.train_data)
            regularization = 0.5 * self.ro * torch.norm(self.localPCA - self.localZ + 1/self.ro * self.localLamda) ** 2
            self.loss = torch.norm(residual, p="fro") ** 2
            print("self.loss", self.loss.data)
            self.lossADMM = self.loss + regularization
            print("self.lossADMM", self.lossADMM)
            temp = self.localPCA.data.clone()
            # slove local problem locally
            if self.localPCA.grad is not None:
                self.localPCA.grad.data.zero_()

            self.lossADMM.backward(retain_graph=True)
            #grad = torch.autograd.grad(self.lossADMM, self.localPCA)
            localGrad = self.localPCA.grad.data.clone()# grad[0]
            # update local pca
            temp  = temp - self.learning_rate * localGrad
            self.localPCA = temp.data.clone()
        return 1