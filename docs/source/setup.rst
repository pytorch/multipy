Installation
============

Building ``torch::deploy`` via Docker
-------------------------------------

The easiest way to build ``torch::deploy``, along with fetching all interpreter
dependencies, is to do so via docker.

.. code:: shell

   git clone https://github.com/pytorch/multipy.git
   cd multipy
   export DOCKER_BUILDKIT=1
   docker build -t multipy .

The built artifacts are located in ``multipy/runtime/build``.

To run the tests:

.. code:: shell

   docker run --rm multipy multipy/runtime/build/test_deploy

Installing via ``pip install``
------------------------------

We support installing both the python modules and the c++ bits (through ``CMake``)
using a single ``pip install -e .`` command, with the caveat of having to manually
install the dependencies first.

First clone multipy and update the submodules:

.. code:: shell

   git clone https://github.com/pytorch/multipy.git
   cd multipy
   git submodule sync && git submodule update --init --recursive

Installing system dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The runtime system dependencies are specified in ``build-requirements-{debian,centos8}.txt``.
To install them on Debian-based systems, one could run:

.. code:: shell

   sudo apt update
   xargs sudo apt install -y -qq --no-install-recommends < build-requirements-debian.txt

While to install on a CentOS 8 system:

.. code:: shell

   xargs sudo dnf install -y < build-requirements-centos8.txt

Installing environment encapsulators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend using the isolated python environments of either `conda
<https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html#regular-installation>`__
or `pyenv + virtualenv <https://github.com/pyenv/pyenv.git>`__
because ``torch::deploy`` requires a
position-independent version of python to launch interpreters with. For
``conda`` environments we use the prebuilt ``libpython-static=3.x``
libraries from ``conda-forge`` to link with at build time. For
``virtualenv``/``pyenv``, we compile python with the ``-fPIC`` flag to create the
linkable library.

.. warning::
   While `torch::deploy` supports Python versions 3.7 through 3.10,
   the ``libpython-static`` libraries used with ``conda`` environments
   are only available for ``3.8`` onwards. With ``virtualenv``/``pyenv``
   any version from 3.7 through 3.10 can be
   used, as python can be built with the ``-fPIC`` flag explicitly.

Installing pytorch and related dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``torch::deploy`` requires the latest version of pytorch to run models
successfully, and we recommend fetching the latest *nightlies* for
pytorch and also cuda.

Installing the python dependencies in a ``conda`` environment:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: shell

   conda create -n newenv
   conda activate newenv

   conda install python=3.8 # or 3.8/3.10
   conda install -c conda-forge libpython-static=3.8 # or 3.8/3.10

   # install your desired flavor of pytorch from https://pytorch.org/get-started/locally/
   conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly

Installing the python dependencies in a  ``pyenv`` / ``virtualenv`` setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: shell

   # feel free to replace 3.8.6 with any python version > 3.7.0
   export CFLAGS="-fPIC -g"
   ~/.pyenv/bin/pyenv install --force 3.8.6
   virtualenv -p ~/.pyenv/versions/3.8.6/bin/python3 ~/venvs/multipy
   source ~/venvs/multipy/bin/activate
   pip install -r dev-requirements.txt

   # install your desired flavor of pytorch from https://pytorch.org/get-started/locally/
   pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

Running ``pip install``
~~~~~~~~~~~~~~~~~~~~~~~

Once all the dependencies are successfully installed,
including a ``-fPIC`` enabled build of python and the latest nightly of pytorch, we
can run the following, in either ``conda`` or ``virtualenv``, to install
both the python modules and the runtime/interpreter libraries:

.. code:: shell

   # from base torch::deploy directory
   pip install -e .
   # alternatively one could run
   python setup.py develop

The C++ binaries should be available in ``/opt/dist``.

Alternatively, one can install only the python modules without invoking
``cmake`` as follows:

.. code:: shell

   # from base multipy directory
   pip install  -e . --install-option="--cmakeoff"

.. warning::
   As of 10/11/2022 the linking of prebuilt static ``-fPIC``
   versions of python downloaded from ``conda-forge`` can be problematic
   on certain systems (for example Centos 8), with linker errors like
   ``libpython_multipy.a: error adding symbols: File format not recognized``.
   This seems to be an issue with ``binutils``, and `these steps
   <https://wiki.gentoo.org/wiki/Project:Toolchain/Binutils_2.32_upgrade_notes/elfutils_0.175:_unable_to_initialize_decompress_status_for_section_.debug_info>`__
   can help. Alternatively, the user can go with the
   ``virtualenv``/``pyenv`` flow above.

Running ``torch::deploy`` build steps from source
-------------------------------------------------

Both ``docker`` and ``pip install`` options above are wrappers around
the cmake build of `torch::deploy`. If the user wishes to run the
build steps manually instead, as before the dependencies would have to
be installed in the user’s (isolated) environment of choice first. After
that the following steps can be executed:

Building
~~~~~~~~

.. code:: bash

   # checkout repo
   git checkout https://github.com/pytorch/multipy.git
   git submodule sync && git submodule update --init --recursive

   cd multipy
   # install python parts of `torch::deploy` in multipy/multipy/utils
   pip install -e . --install-option="--cmakeoff"

   cd multipy/runtime

   # build runtime
   mkdir build
   cd build
   # use cmake -DABI_EQUALS_1=ON .. instead if you want ABI=1
   cmake ..
   cmake --build . --config Release

Running unit tests for ``torch::deploy``
----------------------------------------

We first need to generate the neccessary examples. First make sure your
python enviroment has `torch <https://pytorch.org>`__. Afterwards, once
``torch::deploy`` is built, run the following (executed automatically
for ``docker`` and ``pip`` above):

.. code:: bash

   cd multipy/multipy/runtime
   python example/generate_examples.py
   cd build
   ./test_deploy
