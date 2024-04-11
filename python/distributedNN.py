import os
import sys
import torch
import random
import numpy as np
import subprocess
import math
from skimage.transform import resize
import socket
import traceback
import datetime
from torch.multiprocessing import Process
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils import data
from random import Random
from sklearn.model_selection import train_test_split


import io
import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from hdfs import InsecureClient
import torch.distributed as dist


class Classifier(nn.Module):

    def __init__(self, input_size=6, hidden_size=25, output_size=2):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


class DataPartitioner(object):

    def __init__(self, df):
        self.df = df
        self.partitions = np.split(df, dist.get_world_size())
        for partition in self.partitions:
            print(type(partition), partition)

    def use(self, rank):
        return self.partitions[rank]


def impute(data):
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data)
    return pd.DataFrame(data_imputed, columns=data.columns)


def categoriesToNumbers(data):
    for i, row in data.iterrows():
        alg = data.at[i,'algae bloom']
        alg_res = 1 if alg == 'yes' else 0
        data.at[i,'algae bloom'] = alg_res

        date = data.at[i, 'week']
        parts = date.split('-')
        date_res = int(parts[1])
        data.at[i,'week'] = date_res


def convertToInt(data):
    data['algae bloom'] = data['algae bloom'].astype(int)


def getFeaturesAndTarget(df, features: list, target: str):
    X = df[features]
    y = df[target]
    return X, y


def getTensors(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor


def formatData(data):
    categoriesToNumbers(data)
    # data = impute(data)
    data = data.dropna()
    convertToInt(data)
    return data


def partition_dataset():
    data_path = '/cs535/termProject/input_data_no_header.csv'
    client = InsecureClient('http://richmond.cs.colostate.edu:30102')
    with client.read(data_path) as reader:
        csv_data = reader.read()
        csv_data = csv_data.decode('utf-8')
        csv_stream = io.StringIO(csv_data)
        csv_reader = csv.reader(csv_stream)
        data_raw = [row for row in csv_reader][1:]
    data_raw = pd.DataFrame(data_raw, columns=['temperature', 'nitrate', 'phosphorus', 'flow', 'ph', 'week', 'algae bloom'])
    data = formatData(data_raw)
    data = data[pd.to_numeric(data['temperature'], errors='coerce').notnull()]
    data = data[pd.to_numeric(data['nitrate'], errors='coerce').notnull()]
    data = data[pd.to_numeric(data['phosphorus'], errors='coerce').notnull()]
    data = data[pd.to_numeric(data['flow'], errors='coerce').notnull()]
    data = data[pd.to_numeric(data['ph'], errors='coerce').notnull()]
    data = data.dropna()
    print(len(data.index), data[:5])

    leftover = len(data.index) % dist.get_world_size()
    drop_indexes = [x for x in range(leftover)]
    data = data.drop(drop_indexes)

    partitioner = DataPartitioner(data)
    partition = partitioner.use(dist.get_rank())
    return partition


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='', printEnd='\r'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s %% %s' % (prefix, bar, percent, suffix), end=printEnd)
    if (iteration == total):
        print()


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def run():
    print('hello world')
    torch.manual_seed(1234)
    partition = partition_dataset()
    print(type(partition), partition[:5])

    features = ['temperature', 'nitrate', 'phosphorus', 'flow', 'ph', 'week']
    target = 'algae bloom'
    X, y = getFeaturesAndTarget(partition, features, target)

    print(f'X type: {type(X)}, y type: {type(y)}')

    X_tensor, y_tensor = getTensors(X, y)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # hyperparameters
    input_size = X_train.shape[1]
    hidden_size = 25  # 5 predictors... no sure what I sure set here
    output_size = 2  # 2 classes "yes" (1), "no" (0)

    # model, loss function, optimizer
    model = Classifier(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    num_epochs = 100
    for epoch in range(num_epochs):

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass & optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    with torch.no_grad():
        model.eval()
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f'Accuracy: {accuracy:.4f}')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'richmond'
    os.environ['MASTER_PORT'] = '17171'
    dist.init_process_group('gloo', rank=int(rank), world_size=int(world_size), init_method="tcp://porsche:23456", timeout=datetime.timedelta(weeks=120))
    torch.manual_seed(42)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Please provide a rank and world size')
    try:
        setup(sys.argv[1], sys.argv[2])
        run()
    except Exception as e:
        traceback.print_exc()
