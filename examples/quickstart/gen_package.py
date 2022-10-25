# An example usage of `torch.package` which is
# used for our quickstart example.

import torchvision
from torch.package import PackageExporter

# Instantiate some model
model = torchvision.models.resnet.resnet18()

# Package and export it.
with PackageExporter("my_package.pt") as e:
    e.intern("torchvision.**")
    e.extern("numpy.**")
    e.extern("sys")
    e.extern("PIL.*")
    e.extern("typing_extensions")
    e.save_pickle("model", "model.pkl", model)
