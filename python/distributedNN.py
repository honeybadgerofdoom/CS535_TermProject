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


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        features = self.transform(self.preprocess_features(self.dataframe.iloc[idx, :-1]))
        label = torch.tensor(self.dataframe.iloc[idx, -1], dtype=torch.long)
        return features, label

    def preprocess_features(self, features):
        print(f'features before\n{features}')
        features = features.fillna(0)
        features = features.astype(float)
        print(f'features after\n{features}')
        return features


class Partition(object):

    def __init__(self, data, partition):
        self.data = data
        self.partition = partition

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        index = len(self.partition) % index
        if index == len(self.partition):
            index -= 1

        data_idx = self.partition[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, partition_size, seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for i in range(dist.get_world_size()):
            start_index = i * partition_size
            end_index = (i + 1) * partition_size
            self.partitions.append(indexes[start_index:end_index])

    def use(self, rank):
        return Partition(self.data, self.partitions[rank])



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


def getFeaturesAndTarget(data, features: list, target: str):
    X = data[features]
    y = data[target]
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

    size = dist.get_world_size()
    bsz = int(128 / float(size))
    partition_size = len(data.index) // size

    dataset = CustomDataset(data)

    partition = DataPartitioner(dataset, partition_size)
    partition = partition.use(dist.get_rank())

    train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True)
    return train_set, bsz


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
    train_set, bsz = partition_dataset()

    if torch.cuda.is_available():
        model = nn.parallel.DistributedDataParallel(Classifier()).float().cuda()
        print('using cuda')
    else:
        model = nn.parallel.DistributedDataParallel(Classifier()).float()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    num_batches = np.ceil(len(train_set.dataset) / float(bsz))
    best_loss = float("inf")
    num_epochs = 10

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        print_progress_bar(0, len(train_set), prefix='Progress: ', suffix='Complete', length=50)
        for i, (data, target) in enumerate(train_set):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
            print_progress_bar(i+1, len(train_set), prefix='Progress: ', suffix='Complete', length=50)
            print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss / num_batches)
            if dist.get_rank() == 0 and epoch_loss / num_batches < best_loss:
                best_loss = epoch_loss / num_batches
                torch.save(model.state_dict(), 'best_model.pth')


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
