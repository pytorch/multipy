// Basic example of using `ReplicatedObject` in `torch::deploy`.
#include <multipy/runtime/deploy.h>
#include <multipy/runtime/path_environment.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  torch::deploy::InterpreterManager m(4);
    // Start an interpreter manager governing 4 embedded interpreters.
    std::shared_ptr<multipy::runtime::Environment> env =
        std::make_shared<multipy::runtime::PathEnvironment>(
            std::getenv("PATH_TO_EXTERN_PYTHON_PACKAGES") // Ensure to set this environment variable (e.g. /home/user/anaconda3/envs/multipy-example/lib/python3.8/site-packages)
        );

  try {
    // Load the model from the torch.package.
    auto I = m.acquireOne();
    auto model_obj = I.global("torch.nn", "Conv2d")({6, 2, 2, 1});
    at::Tensor output = model_obj(inputs).toIValue().toTensor();
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

  } catch (const c10::Error& e) {
    std::cerr << "error creating movable\n";
    std::cerr << e.msg();
    return -1;
  }

  std::cout << "ok\n";
}
