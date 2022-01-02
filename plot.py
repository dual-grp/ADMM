from utils.plot_utils import *
import sys


def main():
    if(sys.argv[1]=="R"): #python plot.py R FAPL 30 OR python plot.py R FGPL 30
        para_name = sys.argv[1]
        algorithm = sys.argv[2]
        dim = int(sys.argv[3])
        outfile = os.path.join("results",para_name + '_'+algorithm+str(dim))
        para_value = [2, 4, 6]

        try:
            f = open(outfile)
            graph( algorithm, para_name,para_value, dim)
        except IOError:
            r_pca(algorithm,dim)
            graph( algorithm, para_name,para_value, dim)
    elif(sys.argv[1]=="rho"): 
        para_name = sys.argv[1]
        algorithm = sys.argv[2]
        dim = int(sys.argv[3])
        outfile = os.path.join("results",para_name + '_'+algorithm+str(dim))
        para_value = [0.1, 1, 10]
        try:
            f = open(outfile)
            graph(algorithm, para_name,para_value, dim)
        except IOError:
            rho_pca(algorithm,dim)
            graph( algorithm, para_name,para_value, dim)
    elif(sys.argv[1]=="eta"): 
        para_name = sys.argv[1]
        algorithm = sys.argv[2]
        dim = int(sys.argv[3])
        outfile = os.path.join("results", para_name + '_'+algorithm+str(dim))
        etas = [0.000001, 0.0000001, 0.00000001]
        etas_cifar_fapl = [0.00000002, 0.000000002, 0.0000000002]
        para_value = [etas, etas_cifar_fapl]
        try:
            f = open(outfile)
            graph(algorithm, para_name,para_value, dim)
        except IOError:
            print("----"* 100)
            eta_pca(algorithm , dim)
            graph( algorithm, para_name,para_value, dim)
    elif(sys.argv[1]=="comparefixed"):
        outfile = os.path.join("results","Compare_fixed30")
        try:
            f = open(outfile)
            graph_compare(True, 30)
        except IOError:
            fixed = True
            dim = 30
            compare_fixed()
            graph_compare(fixed, dim)
    elif(sys.argv[1]=="compareoptimal"):
        outfile = os.path.join("results","Compare_Optimal30")
        try:
            f = open(outfile)
            graph_compare(False, 30)
        except IOError:
            fixed = False
            dim = 30
            compare_unfixed()
            graph_compare(fixed, dim)
    elif(sys.argv[1]=="compare"):
        classification()




if __name__ == '__main__':
   main()

    
            
            
            
                




