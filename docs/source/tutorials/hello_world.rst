Hello World (Using InterpreterSession directly)
===============================================

Here we use ``torch::deploy`` to print ``Hello World`` to the console
without using ``torch.package``. Instead we simply acquire an individual
``InterpreterSession``, and use it to print ``Hello World`` directly.

Printing Hello World with ``torch::deploy``
-------------------------------------------

.. literalinclude:: ../../../examples/hello_world/hello_world_example.cpp
   :language: c++
   :lines: 5-

Here we introduce the pythonic nature of ``torch::deploy``\ ’s
``InterpreterSession``s by using them in a similar fasion to
python objects. This allows us to add further flexibility to the code
exported by ``torch.package`` by interacting with it in C++.

``manager.acquireOne`` allows us to create an individual subinterpreter
we can interact with.

``InterpreterSession::global(const char* module, const char* name)``
allows us to access python modules such as ``builtins`` and their
attributes such as ``print``. This function outputs an ``Obj``
which is a wrapper around ``print``. From here we call ``print`` by
using ``{"Hello world!"}`` as its argument(s).

Build and execute
-----------------

Assuming the above C++ program was stored in a file called
``hello_world_example.cpp``, a minimal ``CMakeLists.txt`` file would
look like:

.. literalinclude:: ../../../examples/CMakeLists.txt
   :language: cmake
   :lines: 1-20,21,22

From here we execute the hello world program

.. code:: bash

   mkdir build
   cd build
   cmake -S . -B build/ -DMULTIPY_PATH="<Path to Multipy Library>" -DPython3_EXECUTABLE="$(which python3)" && \
   cmake --build build/ --config Release -j
   ./hello_world
