import torch

from FLAlgorithms.users.userGrassmannPCA import UserGrassmannPCA
from FLAlgorithms.servers.serverbaseGrassmann import ServerGrassmann
from FLAlgorithms.trainmodel.models import *
from utils.model_utils import read_data, read_user_data
import numpy as np

class Grassmann(ServerGrassmann):
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
                #self.commonPCAz = V
                print("type of V", type(V))
                print("shape of V: ", V.shape)
                self.commonPCAz = torch.rand_like(V, dtype=torch.float)
                print(self.commonPCAz)
                check = torch.matmul(V.T,V)
            
            user = UserGrassmannPCA(device, id, train, test,y_train, y_test , self.commonPCAz, learning_rate, ro, local_epochs, dim)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")

    def train(self):
        losses = []
        self.selected_users = self.select_users(1000,1)
        print("Selected users: ")
        for user in self.selected_users:
            print("user_id: ", user.id)
        for glob_iter in range(self.num_glob_iters):
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: ",glob_iter, " -------------")
            #loss_ = 0
            self.send_pca()

            # Evaluate model each interation
            loss = self.evaluate()
            losses.append(loss.item())
            # self.selected_users = self.select_users(glob_iter,self.num_users)
            
            # self.users = self.selected_users 
            #NOTE: this is required for the ``fork`` method to work
            for user in self.selected_users:
                user.train(self.local_epochs)
            # self.users[0].train(self.local_epochs)
            self.aggregate_pca()
        Z = self.commonPCAz.detach().numpy()
        np.save('Grassmann_ADMM_3components_Random', Z)
        print("Completed training!!!")
        print(losses)
        return losses
        # self.save_results()
        # self.save_model()
    
    def test(self, model, dataset, global_epoch):
        # Get device status: Check GPU or CPU
        device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
        if(dataset == "Mnist" or dataset == "Cifar10"):
            model = DNN2(self.dim, 100,100,10).to(device), model
        if(dataset == "Synthetic"):
            model = DNN2(self.dim,20,20,10).to(device), model

        accuracy = 0
        for user in self.users: 
           accuracy += user.test(model, dataset, global_epoch, 2)
        accuracy = accuracy / len(self.users)
        print("------ averaged accuracy ",accuracy,"---------")
        return accuracy
          

        