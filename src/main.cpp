#include <filesystem>
#include <iostream>

#include <cxxopts.hpp>

#include "emitter.hpp"
#include "parser.hpp"

namespace fs = std::filesystem;
namespace Opts = cxxopts;

int main(int argc, char **argv) {

  bool help;
  fs::path config;
  std::vector<std::string> cuda_flags;

  Opts::Options options("solver-gen");
  options.add_options()(
      "h,help", "Show available options",
      Opts::value(help)->default_value("false")->implicit_value("true"))(
      "c,config", "Path to config.yml",
      Opts::value(config)->default_value("config.yml"))(
      "f,flags", "Flags passed to CUDA toolchain", Opts::value(cuda_flags));
  options.parse(argc, argv);

  if (help) {
    std::cout << options.help() << '\n';
    return 0;
  }

  try {

    auto config_dir = RKI::Emitter::configureDirectory(config);

    auto parsed_obj = RKI::Parser::parse(config);

    RKI::Emitter::emitSrcs(config_dir, parsed_obj);

    RKI::Emitter::emitExe(config_dir, cuda_flags);

  } catch (std::exception &e) {
    std::cerr << e.what() << '\n';
    return 0;
  }

  return 0;
}