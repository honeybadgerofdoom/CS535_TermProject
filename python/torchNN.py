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
        alg = data.at[i,'algae bloom']
        alg_res = 1 if alg == 'yes' else 0
        data.at[i,'algae bloom'] = alg_res

        date = data.at[i, 'week']
        parts = date.split('-')
        date_res = int(parts[1])
        data.at[i,'week'] = date_res


def getData(input_path):
    data = pd.read_csv(
        input_path,
        skiprows=1,
        usecols=[0,1,2,3,4,5,6],
        names=['temperature', 'nitrate', 'phosphorus', 'flow', 'ph', 'week', 'algae bloom']
    )
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
    # data = impute(data)
    data = data.dropna()
    convertToInt(data)
    return data



def calculate_metrics(y_true, y_pred):
    TP = ((y_true == 1) & (y_pred == 1)).sum().item()
    FP = ((y_true == 0) & (y_pred == 1)).sum().item()
    FN = ((y_true == 1) & (y_pred == 0)).sum().item()

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1_score


def train():

    data_raw = getData('../data/input_data.csv')
    data = formatData(data_raw)

    count_1 = (data['algae bloom'] == 1).sum()
    df_0 = data[data['algae bloom'] == 0].sample(n=count_1, random_state=42)
    df_1 = data[data['algae bloom'] == 1]
    balanced_df = pd.concat([df_0, df_1])
    data = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

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

        precision, recall, f1_score = calculate_metrics(y_test, predicted)
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')


if __name__ == "__main__":
    train()
