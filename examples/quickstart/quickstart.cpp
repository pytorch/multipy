/* quickstart.cpp highlights some of the most basic concepts of `torch::deploy`.
Specifically loading a pytorch model which was serialized by `torch.package`
and running it.

In order to run this file, one needs to provide an archive produced by
`torch.package`. The one used in our example is created by `gen_package.py`
which produces my_package.pt. This program takes the path to the archive as
an argument.
*/

#include <multipy/runtime/deploy.h>
#include <multipy/runtime/path_environment.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  // Start an interpreter manager governing 4 embedded interpreters.
  torch::deploy::InterpreterManager manager(4);
  torch::deploy::ReplicatedObj model;
  try {
    // Load the model from the multipy.package.
    torch::deploy::Package package = manager.loadPackage(argv[1]);
    model = package.loadPickle("model", "model.pkl");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    std::cerr << e.msg();
    return -1;
  }

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}));

  // Execute the model and turn its output into a tensor.
  at::Tensor output = model(inputs).toTensor();
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

  std::cout << "ok\n";
}
