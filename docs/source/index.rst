:github_url: https://github.com/pytorch/multipy

``torch::deploy`` [Beta]
=====================

``torch::deploy``  (MultiPy for non-PyTorch use cases) is a C++ library that enables you to run eager mode PyTorch models in production without any modifications to your model to support tracing. ``torch::deploy`` provides a way to run using multiple independent Python interpreters in a single process without a shared global interpreter lock (GIL).
For more information on how ``torch::deploy`` works please see the related `arXiv paper <https://arxiv.org/pdf/2104.00254.pdf>`_.


Documentation
---------------

.. toctree::
   :maxdepth: 2
   :caption: Usage

   setup.md
   tutorials/tutorial_root
   api/library_root

Acknowledgements
----------------

This documentation website for the MultiPy C++ API has been enabled by the
`Exhale <https://github.com/svenevs/exhale/>`_ project and generous investment
of time and effort by its maintainer, `svenevs <https://github.com/svenevs/>`_.
We thank Stephen for his work and his efforts providing help with both the PyTorch and MultiPy C++ documentation.
