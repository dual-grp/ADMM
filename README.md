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
    <pre><code>
    python plot.py R FAPL 30 
    python plot.py R FGPL 30
    </code></pre>
  - Effects of step size for dual variables, run below commands:
    <pre><code>
    python plot.py rho FAPL 30 
    python plot.py rho FGPL 30
    </code></pre>
  - Effects of local learning rates, run below commands:
    <pre><code>
    python plot.py eta FAPL 30 
    python plot.py eta FGPL 30
    </code></pre>
## Compare FAPL, FGPL and Centralised-PCA
- To produce the comparison experiments on FAPL, FGPL and Centralised-PCA using **fixed hyperparameters**

    <pre><code>
    python plot.py comparefixed
    </code></pre>
- To produce the comparison experiments on FAPL, FGPL and Centralised-PCA using **opyimal hyperparameters**

    <pre><code>
    python plot.py compareoptimal
    </code></pre>
  
