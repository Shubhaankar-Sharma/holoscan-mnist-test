import torch
import torch.onnx
import torch.nn as nn

# Define the model (same architecture as during training)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        self.fc = nn.Linear(5 * 5 * 32, 10)

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("mnist_classifier.pt"))
model.eval()

# Dummy input for export (batch_size=1, channels=1, height=28, width=28)
dummy_input = torch.randn(1, 1, 28, 28).to(device)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "cnn_mnist_model.onnx",
    input_names=["input_tensor"],
    output_names=["output_tensor"],
    dynamic_axes=None,
)

print("Model successfully exported to cnn_mnist_model.onnx")

