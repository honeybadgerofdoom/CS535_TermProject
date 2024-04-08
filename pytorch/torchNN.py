import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


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


def impute(data):
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data)
    return pd.DataFrame(data_imputed, columns=data.columns)


def categoriesToNumbers(data):
    for i, row in data.iterrows():
        val = data.at[i,'algae bloom']
        res = 1 if val == 'yes' else 0
        data.at[i,'algae bloom'] = res


def getData(input_path):
    data = pd.read_csv(input_path)
    data = data.drop(['date'], axis=1)
    return data


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
    data = impute(data)
    convertToInt(data)
    return data


def train():

    data_raw = getData('../data/input_data.csv')
    data = formatData(data_raw)

    features = ['temperature', 'nitrate', 'phosphorus', 'flow', 'ph']
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


if __name__ == "__main__":
    train()
