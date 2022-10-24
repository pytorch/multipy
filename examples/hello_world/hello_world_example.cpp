/* Basic example of a single interpreter from `torch::deploy`
   to invoke python methods directly. Here we specifically
   invoke `print` to print out `Hello World`.
*/
#include <multipy/runtime/deploy.h>
#include <multipy/runtime/path_environment.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  // create two interpreters
  multipy::runtime::InterpreterManager manager(2);

  // Acquire a session on one of the interpreters
  auto I = manager.acquireOne();

  // from builtins import print
  // print("Hello world!")
  I.global("builtins", "print")({"Hello world!"});
}
