#include "deploy.h"
#include "path_environment.h"

namespace torch {
namespace deploy {

void PathEnvironment::configureInterpreter(Interpreter* interp) {
  auto I = interp->acquireSession();
  I.global("sys", "path").attr("append")({path_});
}

} // namespace deploy
} // namespace torch
