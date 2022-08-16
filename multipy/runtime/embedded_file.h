#pragma once

#include <string>

namespace torch {
namespace deploy {

struct ExeSection {
  const char* sectionName;
  bool customLoader;
};

struct InterpreterSymbol {
  const char* startSym;
  const char* endSym;
  bool customLoader;
};

struct EmbeddedFile {
  std::string libraryName{""};
  bool customLoader{false};

  EmbeddedFile(
      std::string name,
      const std::initializer_list<ExeSection>& sections,
      const std::initializer_list<InterpreterSymbol> symbols);

  ~EmbeddedFile();

  EmbeddedFile& operator=(const EmbeddedFile&) = delete;
};

} // namespace deploy
} // namespace torch
