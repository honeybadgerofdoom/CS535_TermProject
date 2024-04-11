import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from hdfs import InsecureClient
import torch.distributed as dist
import datetime
import random
import io
import csv


class Classifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
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


class Partition(object):

    def __init__(self, data, index):
        self.data = date
        self.index = index

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])



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


def train():

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
    print(data[:5])

    '''

    features = ['temperature', 'nitrate', 'phosphorus', 'flow', 'ph', 'week']
    target = 'algae bloom'
    X, y = getFeaturesAndTarget(data, features, target)

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
    '''

    '''
    Distributed training loop
    Make sure to use DistributedSampler for distributed data loading
    Use DistributedDataParallel for distributed training
    Example:
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    train_loader = torch.utils.data.DataLoader(dataset, sampler=sampler)
    model = torch.nn.parallel.DistributedDataParallel(model)
    Also, make sure to adjust the batch size accordingly for distributed training
    '''

    '''
    # TRAINING
    num_epochs = 100
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # EVAL
    with torch.no_grad():
        model.eval()
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f'Accuracy: {accuracy:.4f}')
        
    '''

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'richmond'
    os.environ['MASTER_PORT'] = '17171'
    dist.init_process_group('gloo', rank=int(rank), world_size=int(world_size), init_method="tcp://porsche:23456", timeout=datetime.timedelta(weeks=120))
    torch.manual_seed(42)


if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print('Please provide a rank and world size')
    # try:
    #     setup(sys.argv[1], sys.argv[2])
        train()
    # except Exception as e:
    #     traceback.print_exc()
