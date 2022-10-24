// Basic example of using `ReplicatedObject` in `torch::deploy`.
#include <multipy/runtime/deploy.h>
#include <multipy/runtime/path_environment.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  torch::deploy::InterpreterManager m(4);

  try {
    // Load the model from the torch.package.
    auto I = m.acquireOne();
    std::vector<torch::jit::IValue> constructor_inputs;
    auto model_obj = I.global("torch.nn", "Conv2d")({6, 2, 2, 1});
    auto rObj = m.createMovable(model_obj, &I);
    auto I2 = m.acquireOne();
    auto model_obj2 = I2.fromMovable(rObj);
    rObj.unload(); // free the replicated object

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 6, 6, 6}));

    // Execute the model and turn its output into a tensor.
    at::Tensor output = model_obj2(inputs).toIValue().toTensor();
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

  } catch (const c10::Error& e) {
    std::cerr << "error creating movable\n";
    std::cerr << e.msg();
    return -1;
  }

  std::cout << "ok\n";
}
