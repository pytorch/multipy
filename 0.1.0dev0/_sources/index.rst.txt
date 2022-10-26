:github_url: https://github.com/pytorch/multipy

``torch::deploy`` [Beta]
=====================

``torch::deploy`` is a system that allows you to load multiple python interpreters which execute PyTorch models, and run them in a single C++ process. Effectively, it allows people to multithread their pytorch models.
For more information on how torch::deploy works please see the related `arXiv paper <https://arxiv.org/pdf/2104.00254.pdf>`_. We plan to further generalize ``torch::deploy`` into a more generic system, ``multipy::runtime``,
which is more suitable for arbitrary python programs rather than just pytorch applications.


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
