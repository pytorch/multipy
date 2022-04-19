file(REMOVE_RECURSE
  "libtorch_python_static.a"
  "libtorch_python_static.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/torch_python_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
