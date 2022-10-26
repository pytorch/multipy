Passing Python Objects from one Interpreter to Another
======================================================

Here we use ``torch::deploy`` to create a `ReplicatedObj` in order to pass
an `Obj` from one interpreter to another.

Moving Python Objects Between Interpreters
------------------------------------------

.. literalinclude:: ../../../examples/movable_example/movable_example.cpp
   :language: c++

Here we highlight ``torch::deploy::InterpreterManager::create_movable(Obj, InterpreterSession*)``
and ``InterpreterSession::fromMovable(const ReplicatedObj&)``. These functions allow conversions
between ``Obj`s` which are speciifc to an interpreter and ``ReplicatedObj``s
which are replicated across multiple interpreters.

Build and execute
-----------------

Assuming the above C++ program was stored in a file called
``movable_example.cpp``, a minimal ``CMakeLists.txt`` file would
look like:

.. literalinclude:: ../../../examples/CMakeLists.txt
   :language: cmake
   :lines: 1-20,27,28

From here we execute the hello world program

.. code:: bash

   mkdir build
   cd build
   cmake -S . -B build/ -DMULTIPY_PATH="<Path to Multipy Library>" -DPython3_EXECUTABLE="$(which python3)" && \
   cmake --build build/ --config Release -j
   ./hello_world
