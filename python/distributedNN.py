import os
import sys
import torch
import numpy as np
import traceback
import datetime
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


import io
import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from hdfs import InsecureClient
import torch.distributed as dist

predictor_combinations = {
    'ALL': ['temperature', 'nitrate', 'phosphorus', 'flow', 'ph', 'week'],
    'CHEMICALS': ['nitrate', 'phosphorus', 'ph',],
    'NUTRIENTS': ['nitrate', 'phosphorus',],
    'ENVIRONMENTAL': ['temperature', 'flow'],
    'NO_TIME': ['temperature', 'nitrate', 'phosphorus', 'flow', 'ph'],
    'TEMPERATURE': ['temperature']
}

class ClassifierSimple(nn.Module):
    def __init__(self, input_size=6, output_size=2, hidden_size=25):
        super(ClassifierSimple, self).__init__()
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

class Classifier(nn.Module):

    def __init__(self, input_size=6, output_size=2, hidden_size1=250, hidden_size2=125, dropout_prob=0.5):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out

class MoreComplexClassifier(nn.Module):

    def __init__(self, input_size=6, output_size=2, hidden_size1=100, hidden_size2=50, hidden_size3=25, dropout_prob=0.5):
        super(MoreComplexClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc4(out)
        out = self.softmax(out)
        return out


classifiers = [
    {'name': 'Simple', 'cl': ClassifierSimple},
    {'name': 'General', 'cl': Classifier},
    {'name': 'Complex', 'cl': MoreComplexClassifier},
]


class DataPartitioner(object):

    def __init__(self, df):
        self.df = df
        self.partitions = np.split(df, dist.get_world_size())

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
    y_tensor = torch.tensor(y.tolist(), dtype=torch.long)
    return X_tensor, y_tensor


def formatData(data):
    categoriesToNumbers(data)
    convertToInt(data)
    return data


def partition_dataset(predictors):
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
    for predictor in predictors:
        data = data[pd.to_numeric(data[predictor], errors='coerce').notnull()]

    count_1 = (data['algae bloom'] == 1).sum()
    df_0 = data[data['algae bloom'] == 0].sample(n=count_1, random_state=42)
    df_1 = data[data['algae bloom'] == 1]
    balanced_df = pd.concat([df_0, df_1])
    data = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

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


def calculate_metrics(y_true, y_pred):
    TP = ((y_true == 1) & (y_pred == 1)).sum().item()
    FP = ((y_true == 0) & (y_pred == 1)).sum().item()
    FN = ((y_true == 1) & (y_pred == 0)).sum().item()

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1_score


def run():
    torch.manual_seed(1234)

    for key in predictor_combinations:
        predictors = predictor_combinations[key]
        partition = partition_dataset(predictors)

        target = 'algae bloom'
        X, y = getFeaturesAndTarget(partition, predictors, target)


        X_tensor, y_tensor = getTensors(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

        input_size = X_train.shape[1]
        output_size = 2

        for classifier in classifiers:

            model_name = classifier['name']
            model = classifier['cl'](input_size, output_size)
            criterion = nn.CrossEntropyLoss()
            sgd = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            adam = optim.Adam(model.parameters(), lr=0.001)

            optimizers = [{'name': 'SGD', 'opt': sgd}, {'name': 'ADAM', 'opt': adam}]

            num_epochs = 1000

            for optimizer in optimizers:
                for epoch in range(num_epochs):

                    outputs = model(X_train)
                    loss = criterion(outputs, y_train)

                    optimizer['opt'].zero_grad()
                    loss.backward()
                    optimizer['opt'].step()

                with torch.no_grad():
                    model.eval()
                    outputs = model(X_test)
                    _, predicted = torch.max(outputs, 1)
                    accuracy = (predicted == y_test).sum().item() / y_test.size(0)

                    precision, recall, f1_score = calculate_metrics(y_test, predicted)
                    optName = optimizer['name']
                    print(f'Mode: {model_name}, Predictors: {key}, Optimizer: {optName}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')


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
