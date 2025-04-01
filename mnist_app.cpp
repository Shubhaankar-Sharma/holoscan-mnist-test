#include <holoscan/holoscan.hpp>
#include <torch/torch.h>
#include <memory>
#include <vector>
#include "inference.hpp"
#include "profiler.hpp" 
#include <nvtx3/nvtx3.hpp>  // Add NVTX header
#include <gxf/std/tensor.hpp>

using holoscan::Operator;
using holoscan::OperatorSpec;
using holoscan::InputContext;
using holoscan::OutputContext;
using holoscan::ExecutionContext;
using holoscan::Arg;
using holoscan::ArgList;

// Define a domain for the MNIST application - separate from our profiler domain
struct mnist_app_domain { static constexpr char const* name{"MNIST_APP"}; };

class MNISTInputOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MNISTInputOp)
  MNISTInputOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<holoscan::gxf::Entity>("input_tensor");
    spec.param(allocator_, "allocator_", "Allocator", "Output Allocator");
  }

  void initialize() override {
    // Use NVTX3 range for initialization - this is stack allocated
    nvtx3::scoped_range_in<mnist_app_domain> init_range{"MNISTInputOp::initialize"};
    
    auto data_path_ = "/home/holoscan/test_app_2/data/MNIST/raw";
    auto base_dataset = torch::data::datasets::MNIST(data_path_);
    auto size_before = base_dataset.size();
    HOLOSCAN_LOG_INFO("Base dataset size: {}", size_before.has_value() ? size_before.value() : -1);
    
    // Apply normalize transform
    auto normalized_dataset = base_dataset.map(torch::data::transforms::Normalize<>(0.1307, 0.3081));
    auto size_after_norm = normalized_dataset.size();
    HOLOSCAN_LOG_INFO("Size after normalize: {}", size_after_norm.has_value() ? size_after_norm.value() : -1);
    
    // Apply stack transform
    auto stacked_dataset = normalized_dataset.map(torch::data::transforms::Stack<>());
    auto size_after_stack = stacked_dataset.size();
    HOLOSCAN_LOG_INFO("Size after stack: {}", size_after_stack.has_value() ? size_after_stack.value() : -1);
    
    dataset_size_ = size_after_stack.value();
    HOLOSCAN_LOG_INFO("Final dataset size: {}", dataset_size_);
    num_processed_ = 0;
    
    // Additional debug info
    try {
        auto example = base_dataset.get(0);
        HOLOSCAN_LOG_INFO("Successfully got first example from base dataset");
        HOLOSCAN_LOG_INFO("Example data size: {}", example.data.sizes());
        HOLOSCAN_LOG_INFO("Example target size: {}", example.target.sizes());
    } catch (const c10::Error& e) {
        HOLOSCAN_LOG_ERROR("Error accessing first example from base dataset: {}", e.what());
    }
    
    Operator::initialize();
  }
  
  void start() override {
    HOLOSCAN_LOG_TRACE("MNISTInputOp::start()");
    // Mark the start event in NVTX
    nvtx3::mark_in<mnist_app_domain>("MNISTInputOp::start");
  }

  template <std::size_t N, std::size_t C>
  void add_data(holoscan::gxf::Entity& entity, const char* name,
                const std::array<std::array<float, C>, N>& data, ExecutionContext& context) {
    // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                         allocator_->gxf_cid());
    // add a tensor
    auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>(name).value();
    // reshape the tensor to the size of the data
    tensor->reshape<float>(
        nvidia::gxf::Shape({N, C}), nvidia::gxf::MemoryStorageType::kHost, allocator.value());
    // copy the data to the tensor
    std::memcpy(tensor->pointer(), data.data(), N * C * sizeof(float));
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext& context) override {
    // Create NVTX range for the entire compute method - stack allocated
    nvtx3::scoped_range_in<mnist_app_domain> compute_range{"MNISTInputOp::compute"};
    
    if (num_processed_ == -1) {
      return;
    }

    if (num_processed_ >= dataset_size_) {
      HOLOSCAN_LOG_INFO("MNISTInputOp::Dataset iteration complete");
      HOLOSCAN_LOG_INFO("whY??? Final dataset size: {}", dataset_size_);
      num_processed_ = -1;
      return;
    }

    // Create a stack-allocated range for dataset loading
    nvtx3::scoped_range_in<mnist_app_domain> dataset_range{"MNISTInputOp::load_dataset"};
    
    auto dataset = torch::data::datasets::MNIST("/home/holoscan/test_app_2/data/MNIST/raw")
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());
        
    auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(dataset), 1);

    size_t current = 0;
    for (auto& batch : *loader) {
      if (current++ < num_processed_) {
            continue;
      }
      auto images = batch.data;

      // Mark tensor reshaping in NVTX
      nvtx3::mark_in<mnist_app_domain>("Reshaping tensor");
      
      images = images.reshape({1, 1, 28, 28});

      auto entity = holoscan::gxf::Entity::New(&context);
      auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                      allocator_->gxf_cid());
      // Add tensor to entity
      auto tensor = static_cast<nvidia::gxf::Entity&>(entity)
                .add<nvidia::gxf::Tensor>("input_tensor").value();

      // Create a stack-allocated range for CUDA transfer
      nvtx3::scoped_range_in<mnist_app_domain> cuda_range{"CUDA transfer"};
      
      // Move PyTorch tensor to GPU first
      auto cuda_images = images.to(torch::kCUDA);

      // Reshape tensor with MNIST dimensions - Use kDevice for GPU memory
      tensor->reshape<float>(
          nvidia::gxf::Shape({1, 1, 28, 28}), 
          nvidia::gxf::MemoryStorageType::kDevice,  // Change to GPU memory
          allocator.value());

      // Copy data using CUDA since both tensors are on GPU
      cudaMemcpy(tensor->pointer(), 
                cuda_images.data_ptr<float>(), 
                cuda_images.numel() * sizeof(float), 
                cudaMemcpyDeviceToDevice);

      op_output.emit(entity, "input_tensor");
      num_processed_++;
      break;  // Process only one batch
    }
  }

  void stop() override {
    HOLOSCAN_LOG_TRACE("MNISTInputOp::stop()");
    // Mark the stop event in NVTX
    nvtx3::mark_in<mnist_app_domain>("MNISTInputOp::stop");
  }

  private:
    Parameter<std::shared_ptr<Allocator>> allocator_;
    uint32_t count_ = 0;
    std::string data_path_;
    size_t num_processed_;
    size_t dataset_size_;
};

class PrintOp : public Operator {
  public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PrintOp)
  PrintOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<holoscan::gxf::Entity>("output_tensor");
  }

  void start() override {
    HOLOSCAN_LOG_TRACE("PrintOp::start()");
    // Mark the start event in NVTX
    nvtx3::mark_in<mnist_app_domain>("PrintOp::start");
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override {
    // Create NVTX range for the entire compute method - stack allocated
    nvtx3::scoped_range_in<mnist_app_domain> compute_range{"PrintOp::compute"};
    
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("output_tensor");
    if (maybe_entity) {
      auto& entity = maybe_entity.value();

      auto tensor = entity.get<holoscan::Tensor>("output_tensor");
      if (!tensor) {
        HOLOSCAN_LOG_ERROR("No tensor in output");
        return;
      }

      // Get pointer to raw tensor data
      size_t size = tensor->size();

      // Create a stack-allocated range for CUDA to CPU transfer
      nvtx3::scoped_range_in<mnist_app_domain> cuda_range{"CUDA to CPU transfer"};
      
      // Pointer to GPU data
      const float* gpu_data = static_cast<const float*>(tensor->data());

      // Copy the data to host
      std::vector<float> host_data(size);
      cudaMemcpy(host_data.data(), gpu_data, size*sizeof(float), cudaMemcpyDeviceToHost);

      // Create a stack-allocated range for result processing
      nvtx3::scoped_range_in<mnist_app_domain> processing_range{"Process results"};
      
      // Now 'host_data' is safe to access in CPU code
      // Find predicted digit
      int predicted_digit = 0;
      float max_val = host_data[0];
      for (int i = 1; i < size; ++i) {
        if (host_data[i] > max_val) {
          max_val = host_data[i];
          predicted_digit = i;
        }
      }

      // Softmax (for printing probabilities, if needed)
      float exp_sum = 0.f;
      for (auto& val : host_data) {
        exp_sum += std::exp(val);
      }

      HOLOSCAN_LOG_INFO("Predicted Digit!!!!: {}", predicted_digit);
      HOLOSCAN_LOG_INFO("Confidence: {:.2f}%", 
          (std::exp(host_data[predicted_digit]) / exp_sum) * 100.f);

      // Print all probabilities
      for (int i = 0; i < size; i++) {
        auto prob = (std::exp(host_data[i]) / exp_sum) * 100.f;
        HOLOSCAN_LOG_INFO("Probability for digit {}: {:.2f}%", i, prob);
      }
    }
  }

  void stop() override {
    HOLOSCAN_LOG_TRACE("PrintOp::stop()");
    // Mark the stop event in NVTX
    nvtx3::mark_in<mnist_app_domain>("PrintOp::stop");
  }
};

class MNISTApp : public holoscan::Application {
 public:
  void compose() override {
    // Mark application composition in NVTX - stack allocated
    nvtx3::scoped_range_in<mnist_app_domain> range{"MNISTApp::compose"};
    
    using namespace holoscan;
    
    auto allocator = make_resource<UnboundedAllocator>("pool");
    auto cuda_stream_pool = make_resource<CudaStreamPool>(
        "cuda_stream_pool",
        Arg("dev_id", 0));

    auto input_op = make_operator<MNISTInputOp>(
        "input", 
        Arg("allocator_", allocator));

    auto inference_op = make_operator<InferenceOp>(
        "inference",
        Arg("allocator", allocator),
        Arg("cuda_stream_pool", cuda_stream_pool),
        Arg("backend", std::string("trt")),
        Arg("model_path_map", InferenceOp::DataMap{
            {{"resnet", "/workspace/models/cnn_mnist.onnx"}}
        }),
        Arg("inference_map", InferenceOp::DataVecMap{
            {{"resnet", std::vector<std::string>{"output_tensor"}}}
        }),
        Arg("pre_processor_map", InferenceOp::DataVecMap{
            {{"resnet", std::vector<std::string>{"input_tensor"}}}
        }));

    auto output_op = make_operator<PrintOp>("output");

    add_flow(input_op, inference_op, {{"input_tensor", "receivers"}});
    add_flow(inference_op, output_op, {{"transmitter", "output_tensor"}});
  }
};

int main(int argc, char** argv) {
    // Mark application start in NVTX
    nvtx3::mark_in<mnist_app_domain>("Application Start");
    
    auto app = std::make_unique<MNISTApp>();
    
    {
      // Create a stack-allocated range for the app run
      nvtx3::scoped_range_in<mnist_app_domain> run_range{"Application Run"};
      app->run();
    }
    
    // Mark profiler output in NVTX
    nvtx3::mark_in<mnist_app_domain>("Profiler Report");
    
    HoloscanProfiler::getInstance().printReport();
    HoloscanProfiler::getInstance().saveReportToFile("mnist_inference_profile.csv");
    
    // Mark application end in NVTX
    nvtx3::mark_in<mnist_app_domain>("Application End");
    
    return 0;
}