# Federated Principle Componenet Analysis (PCA) Learning

This repository implements all experiments in the thesis **Federated Principle Componenet Analysis (PCA) Learning**.
  
Author: Huilan Zhu under the supervision of Dr. Nguyen Tran

# Software requirements:
- torch, torchvision, sklearn, pandas, matplotlib, numpy, scipy, tqdm, pillow, H5py.

- To download the dependencies: **pip3 install -r requirements.txt**
  
# Dataset: We use 3 datasets: MNIST, CIFAR-10 and Synthetic
- To generate non-iid MNIST Data: 
  - Access data/Mnist and run: "python3 generate_niid_20users.py"
  - We can change the number of user and number of labels for each user using 2 variable NUM_USERS = 20 and NUM_LABELS = 2

- To generate non-iid CIFAR-10 Data: 
    - Access data/Cifar10 and run: "python3 generate_niid_20users.py"
    - We can change the number of user and number of labels for each user using 2 variable NUM_USERS = 20 and NUM_LABELS = 2
- To generate niid Synthetic:
  - Access data/Synthetic and run: "python3 generate_synthetic_05_05.py". Similar to MNIST data, the Synthetic data is configurable with the number of users and the numbers of labels for each user.

# Produce experiments and figures

- There is a file "plot.py" which allows running all experiments and generate figures


## Effects of hyperparameters
- To produce the experiments on effects of hyperparameters on FAPL and FGPL:

  - Effects of local epochs, run below commands:
  ![R_FAPL30](https://user-images.githubusercontent.com/49133012/147864147-ac9a7e23-5d3b-40c5-8887-1c02ec1bbe54.png)
  ![R_FGPL30](https://user-images.githubusercontent.com/49133012/147864148-1a3d761b-c7bf-4662-8594-7715c4f631fd.png)


    <pre><code>
    python plot.py R FAPL 30 
    python plot.py R FGPL 30
    </code></pre>
  - Effects of step size for dual variables, run below commands:
  ![rho_FAPL30](https://user-images.githubusercontent.com/49133012/147864163-dbca6de0-8c61-426c-b7ab-68ce99a09012.png)
  ![rho_FGPL30](https://user-images.githubusercontent.com/49133012/147864164-7eaaa3ed-80ea-4103-a4b7-09d3d2bdc5ce.png)

    <pre><code>
    python plot.py rho FAPL 30 
    python plot.py rho FGPL 30
    </code></pre>
  - Effects of local learning rates, run below commands:
  ![eta_FAPL30](https://user-images.githubusercontent.com/49133012/147864179-1a9ac552-7dfe-4f7d-b3f1-913b5cb7463c.png)
  ![eta_FGPL30](https://user-images.githubusercontent.com/49133012/147864178-704e7479-e3e6-4c89-9a94-e819f1820e05.png)
    <pre><code>
    python plot.py eta FAPL 30 
    python plot.py eta FGPL 30
    </code></pre>
## Compare FAPL, FGPL and Centralised-PCA
- To produce the comparison experiments on FAPL, FGPL and Centralised-PCA using **fixed hyperparameters**
    <pre><code>
    python plot.py comparefixed
    </code></pre>
    ![Compare_fixed30](https://user-images.githubusercontent.com/49133012/147864060-d7d14993-4dfe-42a4-9d85-906354d7ae1e.png)
- To produce the comparison experiments on FAPL, FGPL and Centralised-PCA using **opyimal hyperparameters**
    <pre><code>
    python plot.py compareoptimal
    </code></pre>
    ![Compare_Optimal30](https://user-images.githubusercontent.com/49133012/147864142-e40c209e-82a5-40ed-bf8a-6843fe6d5386.png)


# Executing FAPL or FGPL with customised parameters
  - The main file "main.py" which allows running a single algorithm on a dataset with specific hyperparameters defined in file "utils/options.py". To run a single algorithm on a specified dataset: 
    <pre><code>
    python3 main.py --dataset Mnist --batch_size 2 --learning_rate 0.00000002 --ro 1 --num_global_iters 20 --local_epochs 4 --dim 30 --optimizer SGD --algorithm FGPL --subusers 20 --times 1
    </code></pre>
    
  
