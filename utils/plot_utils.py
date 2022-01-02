import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from utils.model_utils import read_data
from FLAlgorithms.servers.serverADMM import ADMM
from FLAlgorithms.servers.serverGrassmann import Grassmann
from FLAlgorithms.servers.centralisedPCA import Centralised
torch.manual_seed(0)



font = {'size'   : 12}
plt.rc('font', **font)

def get_training_loss( algorithm, numusers, local_epochs, num_glob_iters, learning_rate, ro, dataset, dim):
    data = read_data(dataset) , dataset
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment = 0
    server = None
    if(algorithm == "FAPL"):
        server = ADMM(experiment, device, data, learning_rate, ro, num_glob_iters, local_epochs, numusers, dim, 1)
    if(algorithm =="FGPL"):
        server = Grassmann(experiment, device, data, learning_rate, ro, num_glob_iters, local_epochs, numusers, dim, 1)
    if(algorithm == "Centralised"):
        server = Centralised(experiment, device, data, learning_rate, ro, num_glob_iters, local_epochs, numusers, dim, 1)
    losses = server.train()

    return losses

# effects of R on Convergence of FAPL/FGPL --> output to r_<algo><dim>
def r_pca(algorithm, dim=30):
    datasets = ["Synthetic", "Mnist", "Cifar10"]
    Rs = (2, 4, 8)
    
    rho = 0.1
    
    loss_list = []
    for dataset in datasets: 
        if(dataset == "Mnist"):
            K = 50
            eta = 0.000001
            numusers = 20
        elif(dataset == "Synthetic"):
            eta = 0.000001
            K = 50
            numusers = 100
        elif(dataset == "Cifar10"):
            eta = 0.00000002
            K = 100
            numusers = 20
        for R in Rs: 
            loss = get_training_loss(algorithm, numusers, R, K, eta , rho, dataset, dim)
            loss_list.append(loss)
    
    outfile = os.path.join("results",'R_'+algorithm+str(dim))
    with open(outfile, 'wb') as fp:
        pickle.dump(loss_list, fp)
    print(loss_list)
    return


# effects of rho on Convergence of FAPL/FGPL --> output to rho_<algo><dim>
def rho_pca(algorithm, dim=30):
    datasets = ["Synthetic", "Mnist", "Cifar10"]
    Rhos = (0.1, 1, 10)
    
    R = 4
    
    loss_list = []
    for dataset in datasets: 
        if(dataset == "Mnist"):
            K = 50
            eta = 0.000001
            numusers = 20
        elif(dataset == "Synthetic"):
            K = 50
            eta = 0.000001
            numusers = 100
        elif(dataset == "Cifar10"):
            K = 100
            eta = 0.00000002
            numusers = 20
        for rho in Rhos: 
            loss = get_training_loss(algorithm, numusers, R, K, eta , rho, dataset, dim)
            loss_list.append(loss)
    
    outfile = outfile = os.path.join("results",'Rho_'+algorithm+str(dim))
    with open(outfile, 'wb') as fp:
        pickle.dump(loss_list, fp)
    print(loss_list)
    return


# effects of eta on Convergence of FAPL/FGPL --> output to eta_<algo><dim>
def eta_pca(algorithm, dim=30):
    datasets = ["Synthetic", "Mnist", "Cifar10"]
    
    etas = [0.000001, 0.0000001, 0.00000001]
    etas_cifar_fapl = [0.00000002, 0.000000002, 0.0000000002]
    R = 4
    rho = 1
    
    loss_list = []
    for dataset in datasets: 
        if(dataset == "Mnist"):
            K = 50
            numusers = 20
        elif(dataset == "Synthetic"):
            K = 50      
            numusers = 100
        elif(dataset == "Cifar10"):
            K = 100 
            numusers = 20
            if(algorithm=="FAPL"):
                etas = etas_cifar_fapl
        for eta in etas: 
            loss = get_training_loss(algorithm, numusers, R, K, eta , rho, dataset, dim)
            loss_list.append(loss)
    
    outfile = outfile = os.path.join("results",'Eta_'+algorithm+str(dim))
    with open(outfile, 'wb') as fp:
        pickle.dump(loss_list, fp)
    print(loss_list)
    return


def compare_fixed():
    datasets = ["Synthetic", "Mnist", "Cifar10"]
    algorithms = ['FAPL', 'FGPL']
    eta = 0.00000002
    R = 4
    rho = 1
    dim = 30

    loss_list = []

    for dataset in datasets: 
        if(dataset == "Mnist"):
            K = 50
            numusers = 20
        elif(dataset == "Synthetic"):
            K = 50      
            numusers = 100
        elif(dataset == "Cifar10"):
            K = 100 
            numusers = 20
        for algorithm in algorithms: 
            loss = get_training_loss(algorithm, numusers, R, K, eta , rho, dataset, dim)
            loss_list.append(loss)
    
    #compute centralised loss
    learning_rate = 0
    ro = 0
    num_glob_iters = 0
    local_epochs = 0
    centralised_losses = []
    for dataset in datasets: 
        data = read_data(dataset) , dataset
        server = Centralised(0, 'cuda' , data, learning_rate, ro, num_glob_iters, local_epochs, numusers, dim, 1)
        loss = server.train()
        centralised_losses.append(loss)

    
    outfile = outfile = os.path.join("results",'Compare_fixed' + str(dim))
    with open(outfile, 'wb') as fp:
        pickle.dump(loss_list, fp)
        pickle.dump(centralised_losses, fp)
    print(loss_list)
    return 


def compare_unfixed():
    datasets = ["Synthetic", "Mnist", "Cifar10"]
    algorithms = ['FAPL', 'FGPL']
    eta = 0.000001
    R = 4
    rho = 1
    dim = 30

    loss_list = []

    for dataset in datasets: 
        for algorithm in algorithms: 
            if(dataset == "Mnist"):
                K = 50
                numusers = 20
            elif(dataset == "Synthetic"):
                K = 50      
                numusers = 100
                if(algorithm == "FGPL"):
                    rho = 0.1
            elif(dataset == "Cifar10"):
                K = 100 
                numusers = 20
                if algorithm == "FAPL":
                    eta = 0.00000002

            loss = get_training_loss(algorithm, numusers, R, K, eta , rho, dataset, dim)
            
            loss_list.append(loss)
    
    #compute centralised loss
    learning_rate = 0
    ro = 0
    num_glob_iters = 0
    local_epochs = 0
    centralised_losses = []
    for dataset in datasets: 
        data = read_data(dataset) , dataset
        server = Centralised(0, 'cuda' , data, learning_rate, ro, num_glob_iters, local_epochs, numusers, dim, 1)
        loss = server.train()
        centralised_losses.append(loss)

    
    outfile = os.path.join("results",'Compare_Optimal' + str(dim))
    with open(outfile, 'wb') as fp:
        pickle.dump(loss_list, fp)
        pickle.dump(centralised_losses, fp)
    print(loss_list)
    return 

def classification():
    datasets = ["Synthetic", "Mnist", "Cifar10"]
    algorithms = ['FAPL', 'FGPL', 'Centralised']
    
    eta = 0.000001
    R = 4
    ro = 1
    dim = 30
    K = 30
    times = 5
    accuracies = []
    for dataset in datasets: 
        for algorithm in algorithms: 
            if(dataset == "Mnist"):
                numusers = 20
            elif(dataset == "Synthetic"):
                numusers = 100
                if(algorithm == "FGPL"):
                    ro = 0.1
            elif(dataset == "Cifar10"):
                numusers = 20
                if algorithm == "FAPL":
                    eta = 0.00000002
            server = None
            for time in range(times): 
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                data = read_data(dataset) , dataset
                if(algorithm == "FAPL"):
                    server = ADMM(0, device, data, eta, ro, K, R, numusers, dim, 1)
                if(algorithm =="FGPL"):
                    server = Grassmann(0, device, data, eta, ro, K, R, numusers, dim, 1)
                if(algorithm == "Centralised"):
                    eta = 0
                    ro = 0
                    K = 0
                    R = 0
                    server = Centralised(0, device, data, eta, ro, K, R, numusers, dim, 1)
                server.train()
                global_epoch = 30
                accur = server.test("dnn", dataset, global_epoch)
                accuracies.append(accur)
    outfile = outfile = os.path.join("results",'performance')
    with open(outfile, 'wb') as fp:
        pickle.dump(accuracies, fp)
    print(accuracies)
    return 

            

def graph( algorithm, parameter_name,parameter_value, dim):
    outfile = os.path.join("results",parameter_name + '_' + algorithm + str(dim))
    pickle_off = open (outfile, "rb")
    losses = pickle.load(pickle_off)

    datasets = ["Synthetic", "Mnist", "Cifar10"]

    linestyles = ['-', '-', '-']
    markers = ["d","p","v"]
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(16,6), gridspec_kw={'width_ratios': [1,1,1]} ) #sharey='row')
    fig.suptitle("Effect of "+ parameter_name +" on Convergence Behaviour of " + algorithm)
    fig.tight_layout()
    lamb = []

    if(parameter_name == "R"):
        lamb = [r': R = ' + r'$2$',r': R = ' + r'$4$',r': R = ' + r'$8$',r': R = ' + r'$2$',r': R = ' + r'$4$',r': R = ' + r'$8$']
    elif(parameter_name == "rho"):
        lamb = [r': $\rho$ = ' + r'$0.1$',r': $\rho$ = ' + r'$1$',r': $\rho$ = ' + r'$10$', r': $\rho$ = ' + r'$0.1$',r': $\rho$ = ' + r'$1$',r': $\rho$ = ' + r'$10$']
    elif(parameter_name == "eta"):
        etas = parameter_value[0]
        etas_cifar_fapl = parameter_value[1]
        lamb = [r': $\eta$ = ' + str(etas[0]), r': $\eta$ = ' + str(etas[1]) , r': $\eta$ = ' + str(etas[2])]
    idx = 0
    for dataset in datasets: 
        parameter_len = len(parameter_value) if len(np.asarray(parameter_value).shape) == 1 else len(parameter_value[0])
            
        for j in range(parameter_len): 
            contain_nan = False
            for x in losses[idx]: 
                if np.isnan(x) == True: 
                    contain_nan = True
                    break
            if(dataset == "Mnist" and not contain_nan):
                ax1.plot(losses[idx][1:], linestyle=linestyles[j], label=algorithm + lamb[j], linewidth  = 1, color=colors[j],marker = markers[j],markevery=0.2, markersize=5)
            elif(dataset == "Synthetic" and not contain_nan):
                ax2.plot(losses[idx][1:], linestyle=linestyles[j], label=algorithm + lamb[j], linewidth  = 1, color=colors[j],marker = markers[j],markevery=0.2, markersize=5)
            elif(dataset == "Cifar10" and not contain_nan):
                if(algorithm == "FAPL" and parameter_name == "eta"):
                    etas_cifar_fapl = parameter_value[1]#etas_cifar_fapl 
                    
                    lamb = [r': $\eta$ = ' + str(etas_cifar_fapl[0]), r': $\eta$ = ' + str(etas_cifar_fapl[1]) , r': $\eta$ = ' + str(etas_cifar_fapl[2])]
                elif(parameter_name == "eta"): 
                    etas = parameter_value[0]
                    lamb = [r': $\eta$ = ' + str(etas[0]), r': $\eta$ = ' + str(etas[1]) , r': $\eta$ = ' + str(etas[2])]
                ax3.plot(losses[idx][1:], linestyle=linestyles[j], label=algorithm + lamb[j], linewidth  = 1, color=colors[j],marker = markers[j],markevery=0.2, markersize=5)
                
            idx += 1
    
    ax1.set_xlabel("Global rounds") 
    ax2.set_xlabel("Global rounds") 
    ax3.set_xlabel("Global rounds") 
    
    ax1.set_ylabel("Global reconstruction loss")  
    
    ax1.title.set_text(datasets[0])
    ax2.title.set_text(datasets[1])
    ax3.title.set_text(datasets[2])

    ax1.legend(loc='upper right', fontsize = 8, framealpha=0.5)
    ax2.legend(loc='upper right', fontsize = 8, framealpha=0.5)
    ax3.legend(loc='upper right', fontsize = 8, framealpha=0.5)
    
    
    outfile_graph =  outfile+".pdf"
    fig.savefig(outfile_graph) #bbox_inches="tight")



def graph_compare(fixed, dim):
    
    outfile = outfile = os.path.join("results","Compare_fixed"+str(dim) if fixed else "Compare_Optimal"+str(dim))
    pickle_off = open (outfile, "rb")
    losses = pickle.load(pickle_off)
    centralised_loss = pickle.load(pickle_off)
    
    datasets = ["Synthetic", "Mnist", "Cifar10"]
    algorithms = ['FAPL', 'FGPL']
    linestyles = ['-', '-', '-', ':']
    markers = ["d","p","v", "o"]
    colors = ['tab:blue', 'r', 'k', 'y', 'darkorange', 'm']
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(16,6), gridspec_kw={'width_ratios': [1,1,1]} ) #sharey='row')
    if(fixed):
        fig.suptitle("Performance Comparison of FAPL, FGPL and centralised PCA with same hyperparameters")
    else: 
        fig.suptitle("Performance Comparison of FAPL, FGPL and centralised PCA with optimal hyperparameters")
    fig.tight_layout()
    idx = 0

    for dataset in datasets: 
        for j in range(len(algorithms)): 
            algorithm = algorithms[j]
            
            contain_nan = False
            for x in losses[idx]: 
                if np.isnan(x) == True: 
                    contain_nan = True
                    break
            if(dataset == "Mnist" and not contain_nan):
                ax1.plot(losses[idx][1:], linestyle=linestyles[j], label=algorithm , linewidth  = 1, color=colors[j],marker = markers[j],markevery=0.2, markersize=5)
            elif(dataset == "Synthetic" and not contain_nan):
                ax2.plot(losses[idx][1:], linestyle=linestyles[j], label=algorithm , linewidth  = 1, color=colors[j],marker = markers[j],markevery=0.2, markersize=5)
            elif(dataset == "Cifar10" and not contain_nan):  
                ax3.plot(losses[idx][1:], linestyle=linestyles[j], label=algorithm , linewidth  = 1, color=colors[j],marker = markers[j],markevery=0.2, markersize=5)
            idx += 1

        if(dataset == "Mnist"): 
            ax1.plot(np.full(50, centralised_loss[0]), linestyle=linestyles[3], label="centralised PCA" , linewidth  = 1, color=colors[3],marker = markers[3],markevery=0.2, markersize=5)
        elif(dataset == "Synthetic" ):
            ax2.plot(np.full(50, centralised_loss[1]), linestyle=linestyles[3], label="centralised PCA" , linewidth  = 1, color=colors[3],marker = markers[3],markevery=0.2, markersize=5)
        elif(dataset == "Cifar10"):
            ax3.plot(np.full(100, centralised_loss[2]), linestyle=linestyles[3], label="centralised PCA" , linewidth  = 1, color=colors[3],marker = markers[3],markevery=0.2, markersize=5)




    ax1.set_xlabel("Global rounds") 
    ax2.set_xlabel("Global rounds") 
    ax3.set_xlabel("Global rounds") 
    
    ax1.set_ylabel("Global reconstruction loss")  
    
    ax1.title.set_text(datasets[0])
    ax2.title.set_text(datasets[1])
    ax3.title.set_text(datasets[2])

    ax1.legend(loc='upper right', fontsize = 8, framealpha=0.5)
    ax2.legend(loc='upper right', fontsize = 8, framealpha=0.5)
    ax3.legend(loc='upper right', fontsize = 8, framealpha=0.5)
    
    
    outfile_graph = outfile+".pdf"
    fig.savefig(outfile_graph) #bbox_inches="tight")



