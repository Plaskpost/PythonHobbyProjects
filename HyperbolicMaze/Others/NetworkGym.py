import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import MiniMap
import DataGenerator


class CurveNet(nn.Module):
    def __init__(self):
        super(CurveNet, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # Linear activation for the output layer
        return x


class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, output, target):
        # Assuming the output and target have the shape [batch_size, 4]
        # where the 4 values are [x1, y1, x2, y2]
        pred_point1 = output[:, :2]  # First predicted point [x1, y1]
        pred_point2 = output[:, 2:]  # Second predicted point [x2, y2]

        target_point1 = target[:, :2]  # First target point [x1, y1]
        target_point2 = target[:, 2:]  # Second target point [x2, y2]

        # Calculate Euclidean distance for both points
        distance1 = torch.sqrt(torch.sum((pred_point1 - target_point1) ** 2, dim=1))
        distance2 = torch.sqrt(torch.sum((pred_point2 - target_point2) ** 2, dim=1))

        # Total distance is the sum of distances for both points
        total_distance = distance1 + distance2

        # Return the mean of the distances (or sum if preferred)
        return total_distance.mean()


def train_model(num_samples):
    # Instantiate the model, define the loss function and the optimizer
    model = CurveNet()
    criterion = EuclideanLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Generate dataset
    X, y = TrainingDataGenerator.generate_data(num_samples)
    X_train, y_train = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    # Step 3: Train the Network
    num_epochs = 20
    batch_size = 32
    prev_loss = 100.

    # Training loop
    for epoch in range(num_epochs):
        model.train()

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        if loss.item() > prev_loss:
            break
        prev_loss = loss.item

    # Save the model
    torch.save(model.state_dict(), '../SavedModels/curve_solver_model.pth')


# Step 4: Predicting New Values
def predict_points(model, parameters):
    model.eval()
    input_data = torch.tensor([parameters], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_data)
    return prediction.numpy()


def load_model(name):
    model = CurveNet()
    model.load_state_dict(torch.load(f'SavedModels/{name}.pth'))
    model.eval()

    return model

def test_model(num_tests):
    model = load_model('curve_solver_model')
    test_input, test_output = TrainingDataGenerator.generate_data(num_tests)

    for i in range(num_tests):
        [p_r, p_phi, N_phi, z] = test_input[i]
        [x1, y1, x2, y2] = test_output[i]
        s1 = (x1, y1)
        s2 = (x2, y2)
        [[x1, y1, x2, y2]] = predict_points(model, [p_r, p_phi, N_phi, z])

        p_cartesian = TrainingDataGenerator.to_cartesian(p_r, p_phi)
        N_cartesian = TrainingDataGenerator.to_cartesian(1., N_phi)
        circle = MiniMap.find_circle(p_cartesian, N_cartesian)

        TrainingDataGenerator.top_down_plot(circle, (x1, y1), (x2, y2), s1, s2)


if __name__ == '__main__':
    #train_model(100000)
    test_model(10)
