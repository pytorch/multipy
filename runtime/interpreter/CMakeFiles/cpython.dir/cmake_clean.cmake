file(REMOVE_RECURSE
  "CMakeFiles/cpython"
  "CMakeFiles/cpython-complete"
  "cpython/bin/python3"
  "cpython/lib/libcrypto.a"
  "cpython/lib/libpython3.8.a"
  "cpython/lib/libpython_stdlib3.8.a"
  "cpython/lib/libssl.a"
  "cpython/src/cpython-stamp/cpython-archive_stdlib"
  "cpython/src/cpython-stamp/cpython-build"
  "cpython/src/cpython-stamp/cpython-configure"
  "cpython/src/cpython-stamp/cpython-download"
  "cpython/src/cpython-stamp/cpython-install"
  "cpython/src/cpython-stamp/cpython-mkdir"
  "cpython/src/cpython-stamp/cpython-patch"
  "cpython/src/cpython-stamp/cpython-update"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/cpython.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
