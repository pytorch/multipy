Quickstart
==========

Packaging a model ``for torch::deploy``
---------------------------------------

``torch::deploy`` can load and run Python models that are packaged with
``torch.package``. You can find ``torch.package``'s documentation
`here <https://pytorch.org/docs/stable/package.html#tutorials>`__.

For now, let’s create a simple model that we can load and run in
``torch::deploy``.

.. literalinclude:: ../../../examples/quickstart/gen_package.py
   :language: python
   :lines: 4-

Note that since "numpy", "sys", "PIL" were marked as "extern",
``torch.package`` will look for these dependencies on the system that
loads this package. They will not be packaged with the model.

Now, there should be a file named ``my_package.pt`` in your working
directory.

Load the model in C++
---------------------

.. literalinclude:: ../../../examples/quickstart/quickstart.cpp
   :language: c++
   :lines: 11-

This small program introduces many of the core concepts of
``torch::deploy``.

An ``InterpreterManager`` abstracts over a collection of independent
Python interpreters, allowing you to load balance across them when
running your code.

Using the ``InterpreterManager::loadPackage`` method, you can load a
``torch.package`` from disk and make it available to all interpreters.

``Package::loadPickle`` allows you to retrieve specific Python objects
from the package, like the ResNet model we saved earlier.

Finally, the model itself is a ``ReplicatedObj``. This is an abstract
handle to an object that is replicated across multiple interpreters.
When you interact with a ``ReplicatedObj`` (for example, by calling
``forward``), it will select an free interpreter to execute that
interaction.

Build and execute the C++ example
---------------------------------

Assuming the above C++ program was stored in a file called
``example-app.cpp``, a minimal ``CMakeLists.txt`` file would look like:

.. literalinclude:: ../../../examples/CMakeLists.txt
   :language: cmake
   :lines: 1-20,24,25

The ``-rdynamic`` and ``--no-as-needed`` flags are needed when linking to the
executable to ensure that symbols are exported to the dynamic table,
making them accessible to the ``torch::deploy`` interpreters (which are dynamically
loaded).

The last step is configuring and building the project. Assuming that our
code directory is laid out like this:

.. code:: shell

   example-app/
       CMakeLists.txt
       quickstart.cpp

We can now run the following commands to build the application from
within the ``example-app/`` folder:

.. code:: bash

   cmake -S . -B build -DMULTIPY_PATH="/home/user/repos/multipy" # the parent directory of multipy (i.e. the git repo)
   cmake --build build --config Release -j

Now we can run our app:

.. code:: bash

   ./example-app /path/to/my_package.pt
