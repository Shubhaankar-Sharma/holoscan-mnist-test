import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import InferenceOp, HolovizOp
from holoscan.resources import UnboundedAllocator, CudaStreamPool
from holoscan.gxf import Entity
import json

try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndi
except ImportError:
    raise ImportError(
        "CuPy must be installed to run this example. See "
        "https://docs.cupy.dev/en/stable/install.html"
    ) from None

class MNISTInputOp(Operator):
    """Operator to load and preprocess MNIST data."""

    def setup(self, spec: OperatorSpec):
        spec.output("input_tensor")

    def initialize(self):
        # Load MNIST dataset with preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),  # Ensure correct size
            transforms.Grayscale(1),      # Ensure single-channel grayscale
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to match training
        ])
        self.dataset = MNIST(root="./data", train=False, download=True, transform=self.transform)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=True)

        # Iterator for the dataset
        self.data_iter = iter(self.loader)

    def compute(self, op_input, op_output, context):
        try:
            input_image, label = next(self.data_iter)
            input_tensor = input_image.cuda()  # Move to GPU
            print(f"Input Label: {label.item()}")  # Print the ground truth label
            print(f"Tensor Shape: {input_tensor.shape}, Type: {input_tensor.dtype}")
            print(f"hmm: {input_image.dtype}")
            op_output.emit(dict({"input_tensor": input_tensor}), "input_tensor")
        except StopIteration:
            print("End of MNIST dataset.")

class PrintOp(Operator):
    """Operator to print and interpret MNIST digit predictions."""

    def setup(self, spec: OperatorSpec):
        spec.input("output_tensor")

    def compute(self, op_input, op_output, context):
        # Receive the output tensor
        holoscan_tensor = op_input.receive("output_tensor")["output_tensor"]

        # Convert Holoscan Tensor to PyTorch Tensor
        output_tensor = torch.tensor(holoscan_tensor, dtype=torch.float32)

        # Find the predicted digit
        predicted_digit = torch.argmax(output_tensor, dim=1).item()
        confidence = torch.softmax(output_tensor, dim=1)[0, predicted_digit].item()

        # Print the result
        print(f"Tensor shape output: {output_tensor.shape}")
        print(f"✅ Predicted Digit: {predicted_digit}")
        print(f"✅ Confidence: {confidence:.4f}")


class InferenceApp(Application):
    def compose(self):
        # Memory allocator
        allocator = UnboundedAllocator(self, name="allocator")

        # CUDA stream pool
        cuda_stream_pool = CudaStreamPool(self, name="cuda_stream", dev_id=0)

        # Input operator for MNIST data
        mnist_input = MNISTInputOp(self, name="mnist_input")

        # Define the inference operator
        inference_op = InferenceOp(
            self,
            name="inference_operator",
            backend="trt",  # TensorRT backend
            allocator=allocator,
            cuda_stream_pool=cuda_stream_pool,
            model_path_map={"resnet": "/workspace/models/cnn_mnist.onnx"},  # ONNX model path
            inference_map={"resnet": ["output_tensor"]},
            pre_processor_map={"resnet": ["input_tensor"]},  # Map MNIST output to model input
        )

        # Visualization operator (Holoviz)
        printer = PrintOp(self, name="printer")

        # Define the data flow
        self.add_flow(mnist_input, inference_op, {("input_tensor", "receivers")})
        self.add_flow(inference_op, printer, {("transmitter", "output_tensor")})


if __name__ == "__main__":
    app = InferenceApp()
    app.run()

