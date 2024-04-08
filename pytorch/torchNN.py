import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np


# Define neural network architecture
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



def train():

    # Load data from CSV
    data_raw = pd.read_csv('../data/input_data.csv')
    data_raw = data_raw.drop(['date'], axis=1)

    for i, row in data_raw.iterrows():
        val = data_raw.at[i,'algae bloom']
        res = 1 if val == 'yes' else 0
        data_raw.at[i,'algae bloom'] = res

    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data_raw)

    data = pd.DataFrame(data_imputed, columns=data_raw.columns)

    data['algae bloom'] = data['algae bloom'].astype(int)

    print(data.head())

    # Split data into features and target
    X = data[['temperature', 'nitrate', 'phosphorus', 'flow', 'ph']]
    y = data['algae bloom']

    # # Convert target variable to numerical format (did this manually)
    # label_encoder = LabelEncoder()
    # y = label_encoder.fit_transform(y)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # Define hyperparameters
    input_size = X_train.shape[1]
    hidden_size = 128
    output_size = 2  # Since there are two classes ("yes" and "no")

    # Initialize the model, loss function, and optimizer
    model = Classifier(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
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

    # Evaluate the model
    with torch.no_grad():
        model.eval()
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f'Accuracy: {accuracy:.4f}')


if __name__ == "__main__":
    print("hello world")
    train()
