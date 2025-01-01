import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class BaseCNN(nn.Module):
    def __init__(self, num_classes, device, num_epochs=100, learning_rate=0.001, patience=10):
        super(BaseCNN, self).__init__()
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience
        self.num_classes = num_classes
        self.device = device

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement this method")
    
    def train_model(self, train_loader, val_loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        best_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(self.num_epochs):
            self.train()
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for inputs, labels in train_loader:                
                inputs = inputs.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

            avg_loss = running_loss / len(train_loader)
            accuracy = correct_predictions / total_samples * 100

            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

            # Run validation every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.eval()
                val_loss = 0.0
                val_correct_predictions = 0
                val_total_samples = 0

                with torch.no_grad():
                    for val_inputs, val_labels in val_loader:
                        val_inputs = val_inputs.cuda()
                        val_labels = val_labels.cuda()

                        val_outputs = self(val_inputs)
                        val_loss += criterion(val_outputs, val_labels).item()

                        _, val_predicted = torch.max(val_outputs.data, 1)
                        val_total_samples += val_labels.size(0)
                        val_correct_predictions += (val_predicted == val_labels).sum().item()

                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = val_correct_predictions / val_total_samples * 100
                print(f'Validation at Epoch [{epoch + 1}/{self.num_epochs}]: Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')

            # Early stopping logic
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0 
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print("Early stopping triggered. Training stopped.")
                break



class BaseCNN1D(BaseCNN):
    def __init__(self, num_classes, device, num_epochs=100, learning_rate=0.001, patience=10):
        super(BaseCNN1D, self).__init__(num_classes, device, num_epochs, learning_rate, patience)
        self.conv1 = None  
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = None  
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Dynamically initialize `conv1` based on input channels
        if self.conv1 is None:
            in_channels = x.size(1)  
            self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=3).to(x.device)
        
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        if self.fc1 is None:
            flattened_size = x.size(1) * x.size(2)
            self.fc1 = nn.Linear(flattened_size, 128).to(x.device)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class BaseCNN2D(BaseCNN):
    def __init__(self, num_classes, device, num_epochs=100, learning_rate=0.001, patience=10):
        super(BaseCNN2D, self).__init__(num_classes, device, num_epochs, learning_rate, patience)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.device = device
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):       
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    
