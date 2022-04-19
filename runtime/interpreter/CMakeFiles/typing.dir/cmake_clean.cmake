file(REMOVE_RECURSE
  "CMakeFiles/typing"
  "CMakeFiles/typing-complete"
  "third_party/typing_extensions.py"
  "typing/src/typing-stamp/typing-build"
  "typing/src/typing-stamp/typing-configure"
  "typing/src/typing-stamp/typing-download"
  "typing/src/typing-stamp/typing-install"
  "typing/src/typing-stamp/typing-mkdir"
  "typing/src/typing-stamp/typing-patch"
  "typing/src/typing-stamp/typing-update"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/typing.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
