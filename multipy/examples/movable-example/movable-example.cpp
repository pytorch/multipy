#include <multipy/runtime/deploy.h>
#include <multipy/runtime/path_environment.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
    if (argc != 1) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    // Start an interpreter manager governing 4 embedded interpreters.
    std::shared_ptr<torch::deploy::Environment> env =
        std::make_shared<torch::deploy::PathEnvironment>(
            std::getenv("PATH_TO_EXTERN_PYTHON_PACKAGES") // Ensure to set this environment variable (e.g. /home/user/anaconda3/envs/multipy-example/lib/python3.8/site-packages)
        );
    torch::deploy::InterpreterManager m(4, env);

    try {
        // Load the model from the multipy.package.
        auto I = m.acquireOne();
        std::vector<torch::jit::IValue> constructor_inputs;
        auto model_obj =
        I.global("torch.nn", "Conv2d")({6, 2, 2, 1});
        auto rObj = I.createMovable(model_obj);
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
