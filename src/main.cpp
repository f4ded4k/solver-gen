#include <cerrno>
#include <filesystem>
#include <iostream>

#include <cxxopts.hpp>

#include "emitter.hpp"
#include "parser.hpp"

namespace fs = std::filesystem;
namespace Opts = cxxopts;

int main(int argc, char **argv) {

  bool help, quiet;
  fs::path config;
  std::vector<std::string> cuda_flags;

  Opts::Options options("solver-gen");
  options.add_options()(
      "h,help", "Show available options",
      Opts::value(help)->default_value("false")->implicit_value("true"))(
      "q,quiet", "Disable output messages",
      Opts::value(quiet)->default_value("false")->implicit_value("true"))(
      "c,config", "Path to config file",
      Opts::value(config)->default_value("config.yml"))(
      "f,flags", "Flags passed to CUDA toolchain", Opts::value(cuda_flags));

  try {
    options.parse(argc, argv);
  } catch (std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }

  if (help) {
    std::cout << options.help() << '\n';
    return 0;
  }

  try {
    if (!quiet)
      std::cout << "Configuring directory structure...\n";
    auto config_dir = RKI::Emitter::configureDirectory(config);

    if (!quiet)
      std::cout << "Parsing config file...\n";
    auto parsed_obj = RKI::Parser::parse(config);

    if (!quiet)
      std::cout << "Emitting source files...\n";
    RKI::Emitter::emitSources(config_dir, parsed_obj);

    if (!quiet)
      std::cout << "Generating executable...\n";
    RKI::Emitter::emitExecutable(config_dir, cuda_flags);

  } catch (std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }

  return 0;
}