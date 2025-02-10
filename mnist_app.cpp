#include <holoscan/holoscan.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/gxf/entity.hpp>
#include <holoscan/core/execution_context.hpp>
#include <torch/torch.h>
#include <gxf/std/tensor.hpp>
#include "inference.hpp"


// MNIST Input Operator
class MNISTInputOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MNISTInputOp)

  MNISTInputOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<nvidia::gxf::Tensor>("input_tensor");
  }

  void initialize() override {
    // No dataset initialization needed for hardcoded input
  }

  void compute(holoscan::InputContext&, holoscan::OutputContext& op_output, holoscan::ExecutionContext& context) override {
    try {
      // Create a hardcoded input tensor
      auto input_tensor = torch::rand({1, 1, 28, 28}, torch::kCUDA); // Random tensor for testing

      // Create GXF tensor
      auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), context.gxf_cid());
      if (!allocator) {
        throw std::runtime_error("Failed to create allocator");
      }

      auto tensor = nvidia::gxf::Tensor::Create(
          allocator.value(),
          nvidia::gxf::Shape{1, 1, 28, 28},
          nvidia::gxf::PrimitiveType::kFloat32,
          nvidia::gxf::MemoryStorageType::kDevice);
      
      if (!tensor) {
        throw std::runtime_error("Failed to create tensor");
      }

      // Copy data from PyTorch tensor to GXF tensor
      cudaMemcpy(tensor.value().data<float>(),
                input_tensor.data_ptr<float>(),
                tensor.value().size(),
                cudaMemcpyDeviceToDevice);
      
      op_output.emit(tensor.value(), "input_tensor");
      
      std::cout << "Emitting hardcoded input tensor" << std::endl;
    } catch (const std::exception& e) {
      std::cout << "Error in compute: " << e.what() << std::endl;
    }
  }
};

// Print Operator
class PrintOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PrintOp)

  PrintOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<nvidia::gxf::Tensor>("tensor");
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext&, holoscan::ExecutionContext&) override {
    auto tensor = op_input.receive<nvidia::gxf::Tensor>("tensor");
    if (!tensor) return;

    // Get tensor data
    auto shape = tensor->shape();
    auto data = tensor->data<float>();

    // Print prediction (assuming output is a 1x10 tensor of class probabilities)
    float max_prob = 0.0f;
    int predicted_class = 0;
    
    for (int i = 0; i < 10; i++) {
      if (data[i] > max_prob) {
        max_prob = data[i];
        predicted_class = i;
      }
    }

    std::cout << "Predicted digit: " << predicted_class 
              << " (confidence: " << max_prob << ")" << std::endl;
  }
};

// Main Application
class MNISTApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    
    // Create operators
    auto mnist_input = make_operator<MNISTInputOp>("mnist_input", 1);
    auto inference = make_operator<ops::InferenceOp>("inference");
    auto print_op = make_operator<PrintOp>("print");
    
    // Define workflow
    add_flow(mnist_input, inference, {{"input_tensor", "input_tensor"}});
    add_flow(inference, print_op, {{"output_tensor", "tensor"}});
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<MNISTApp>();
  
  // Initialize and run
  app->run();

  return 0;
}