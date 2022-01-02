from math import comb
from sklearn.datasets import fetch_openml
from tqdm import trange
import numpy as np
import random
import json
import os

random.seed(1)
np.random.seed(1)
NUM_USERS = 20 # should be muitiple of 10
NUM_LABELS = 2

# Part 1: Obtain MNIST data & Divide by level
# Setup directory for train/test data
train_path = './data/train/mnist_train.json'
test_path = './data/test/mnist_test.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Get MNIST data, normalize, and divide by level
mnist = fetch_openml('mnist_784', data_home='./data')
mu = np.mean(mnist.data.astype(np.float32), 0)
sigma = np.std(mnist.data.astype(np.float32), 0)
mnist.data = (mnist.data.astype(np.float32) - mu)/(sigma+0.001)
mnist_data = []
for i in trange(10):
    idx = mnist.target==str(i)
    mnist_data.append(mnist.data[idx])

print("\nNumb samples of each label:\n", [len(v) for v in mnist_data])
users_lables = []


# Part 2: Allocated client data in a heterogeneous setting in terms of local data sizes and classes

for user in trange(NUM_USERS):
    for j in range(NUM_LABELS):  # 2 labels for each user
        l = (user + j) % 10
        users_lables.append(l)
unique, counts = np.unique(users_lables, return_counts=True)
print("--------------")
print(unique, counts) # unique labels among clients & total number of clients holding each label
#[0 1 2 3 4 5 6 7 8 9] [4 4 4 4 4 4 4 4 4 4]


def ram_dom_gen(total, size):
    """
    generate numbers {n1, ..., n_(size-1)} from total such that sum{n1, ..., n_(size-1)} <= total
    total: total number of observations belongs to a specific label i
    size: number of clients holding observations with label i
    return: a list containing numbers of observations(of label i) to each client e.g.[2441, 1127, 1575, 1760]
    """
    temp = []
    for i in range(size - 1):
        # return random integers from low (inclusive) to high (exclusive).
        val = np.random.randint(total//(size + 1), total//2)  #?
        temp.append(val)
        total -= val
    temp.append(total)
    return temp

# For each label i, distribute observations(of label i) to each users
number_sample = [] # a list of size [number of label, number of clients holding observations from this label]
for total_value, count in zip(mnist_data, counts):
    temp = ram_dom_gen(len(total_value), count) 
    number_sample.append(temp)
print("--------------")
print(number_sample) # a 2d list of length (number of labels). For each label, store data allocation 

# flatten 2d list by column 
i = 0
number_samples = []
for i in range(len(number_sample[0])):
    for sample in number_sample:
        number_samples.append(sample[i])

print("--------------")
print(number_samples)

###### CREATE USER DATA SPLIT #######
# Assign 100 samples to each user
X = [[] for _ in range(NUM_USERS)]
y = [[] for _ in range(NUM_USERS)]
idx = np.zeros(10, dtype=np.int64)
count = 0
print("--------------")
print("user -- label -- current allocated total -- samples belongs to current label")
for user in trange(NUM_USERS):
    for j in range(NUM_LABELS):  # 2 labels for each users
        l = (2*user + j) % 10 
        num_samples =  number_samples[count] # num sample
        count = count + 1
        if idx[l] + num_samples <= len(mnist_data[l]):
            X[user] += mnist_data[l][idx[l]:idx[l]+num_samples].values.tolist()
            y[user] += (l*np.ones(num_samples)).tolist()
            idx[l] += num_samples
            print(user, l, len(X[user]), num_samples)

print("IDX:", idx) # counting samples allocated for each labels

# Part 3: Output allocated client data to json files
# Create data structure
train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
for i in range(NUM_USERS):
    uname = 'f_{0:05d}'.format(i)
    
    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    #print(type(list(zip(*combined))))
    #print(len(list(zip(*combined))))

    X[i][:], y[i][:] = list(zip(*combined))
    num_samples = len(X[i])
    train_len = int(0.75*num_samples)
    test_len = num_samples - train_len
    
    train_data['users'].append(uname) 
    train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
    test_data['num_samples'].append(test_len)

print("Num_samples:", train_data['num_samples'])
print("Total_samples:",sum(train_data['num_samples'] + test_data['num_samples']))
    
with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print("Finish Generating Samples")