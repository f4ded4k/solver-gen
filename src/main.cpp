#include <filesystem>
#include <iostream>

#include "emitter.hpp"
#include "parser.hpp"

namespace fs = std::filesystem;

int main() {

  fs::path config = "./config.yml";

  try {

    auto config_dir = RKI::Emitter::configureDirectory(config);

    auto parsed_obj = RKI::Parser::parse(config);

    RKI::Emitter::emitSrcs(config_dir, parsed_obj);

    // run command nvcc to create executable

  } catch (std::exception &e) {
    std::cerr << e.what() << '\n';
    return 0;
  }

  return 0;
}