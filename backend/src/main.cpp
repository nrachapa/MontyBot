#include <iostream>
#include <fstream>
#include <onnxruntime/onnxruntime_cxx_api.h>

using namespace std;


int main() {
    // Check if model file exists
    ifstream f("../../latest_model.onnx");
    if (!f.good()) {
        cerr << "Error: Model file not found at ../../latest_model.onnx" << endl;
        return 1;
    }
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "../../latest_model.onnx", session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    vector<int64_t> input_tensor_shape{1, 768};
    vector<float> input_tensor_values(input_tensor_shape[1], 0.5f);
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), input_tensor_shape.data(), input_tensor_shape.size()
    );

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    float *output = output_tensors.front().GetTensorMutableData<float>();
    cout << "Model Output: " << *output << endl;
}