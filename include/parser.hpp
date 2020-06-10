#pragma once

#include <algorithm>
#include <filesystem>
#include <sstream>

#include <boost/format.hpp>
#include <yaml-cpp/yaml.h>

#include "grammer.hpp"

namespace RKI {

namespace fs = std::filesystem;

namespace Parser {

using uint16 = uint16_t;
using uint64 = uint64_t;

// Contains parsed configurations
struct ParsedObject {
  uint64 NumSys = 0;
  uint16 NumEq = 0, NumPar = 0, NumIntPar = 0, NumDiff = 2;
  double Stepsize = 0, xBegin = 0, xEnd = 0;

  Gin::symbol x;
  Gin::symtab SymbolTable;

  std::vector<decltype(SymbolTable)::iterator> Parameters;
  std::vector<std::pair<decltype(SymbolTable)::iterator, Gin::ex>> InterParams,
      Equations;
};

namespace detail {

void throwIf(bool expr, const std::string &err_str) {
  if (expr)
    throw std::runtime_error(err_str);
}

auto insertElseThrow(Gin::symtab &table, const std::string &name,
                     const Gin::ex &ex) {
  auto [iter, succ] = table.try_emplace(name, ex);
  throwIf(!succ, "Duplicate identifier detected: " + name);
  return iter;
}

template <typename Callable_>
void traverseArraySpace(const std::string name, const std::vector<uint16> &dims,
                        std::vector<uint16> &idxs, uint16 idx,
                        Callable_ &&callable) {
  if (idx == dims.size()) {
    std::stringstream ss;
    ss << name;
    for (auto &&idx : idxs) {
      ss << '[' << idx << ']';
    }
    callable(ss.str());
    return;
  }
  for (uint16 i = 0; i < dims[idx]; ++i) {
    idxs[idx] = i;
    traverseArraySpace(name, dims, idxs, idx + 1,
                       std::forward<Callable_>(callable));
  }
}

template <typename Parser_, typename Attr_>
auto parseElseThrow(const std::string &text, const Parser_ &parser,
                    Attr_ &attr) {
  auto it = text.begin();
  auto succ = X3::phrase_parse(it, text.end(), parser, X3::space, attr);
  throwIf(!succ || it != text.end(), "Failed to parse " + text);
}

} // namespace detail

ParsedObject parse(const fs::path &config_path) {
  ParsedObject ret;
  auto config_yml = YAML::LoadFile(config_path);
  Grammer::SymbolTable = &ret.SymbolTable;

  // Parse Constants
  ret.NumSys = config_yml["Constants"]["NumSys"].as<decltype(ret.NumSys)>();
  ret.Stepsize =
      config_yml["Constants"]["Stepsize"].as<decltype(ret.Stepsize)>();
  ret.xBegin = config_yml["Constants"]["Begin"].as<decltype(ret.xBegin)>();
  ret.xEnd = config_yml["Constants"]["End"].as<decltype(ret.xEnd)>();

  // Parse independent variable
  ret.x = Gin::symbol("x");
  if (auto var_yml = config_yml["IndependentVar"]; var_yml) {
    std::string attr;
    detail::parseElseThrow(var_yml.as<std::string>(), Grammer::Sym, attr);
    ret.SymbolTable[attr] = ret.x;
  } else {
    ret.SymbolTable["x"] = ret.x;
  }

  // Parse unknown declarations
  auto unknowns = config_yml["UnknownDecls"].as<std::vector<std::string>>();
  std::vector<std::pair<std::string, std::vector<uint16>>> unknown_attrs;
  uint16 total_num_eqs = 0;
  for (auto &&unknown : unknowns) {
    auto &attr = unknown_attrs.emplace_back();
    detail::parseElseThrow(unknown, Grammer::SymArr, attr);
    uint16 array_size = 1;
    for (auto &&dim : attr.second) {
      array_size *= dim;
    }
    total_num_eqs += array_size;
  }
  for (auto &&[name, dims] : unknown_attrs) {
    std::vector<uint16> idxs(dims.size(), 0);
    detail::traverseArraySpace(name, dims, idxs, 0, [&](const std::string &id) {
      auto iter = detail::insertElseThrow(
          ret.SymbolTable, id,
          GinExt::unknown_array(ret.x, ret.NumEq, total_num_eqs));
      ret.NumEq++;
      ret.Equations.emplace_back(iter, Gin::ex(0));
    });
  }

  // Parse parameter declarations
  auto params = config_yml["ParameterDecls"].as<std::vector<std::string>>();
  for (auto &&param : params) {
    std::pair<std::string, std::vector<uint16>> attr;
    auto &[name, dims] = attr;
    detail::parseElseThrow(param, Grammer::SymArr, attr);
    std::vector<uint16> idxs(dims.size(), 0);
    detail::traverseArraySpace(name, dims, idxs, 0, [&](const std::string &id) {
      auto iter = detail::insertElseThrow(ret.SymbolTable, id,
                                          GinExt::param_array(ret.NumPar));
      ++ret.NumPar;
      ret.Parameters.emplace_back(iter);
    });
  }

  // parse intermediate parameter declarations
  if (auto interparams_yml = config_yml["IntermediateParamDecls"];
      interparams_yml) {
    auto interparams = interparams_yml.as<std::vector<std::string>>();
    std::vector<std::pair<std::string, std::vector<uint16>>> interparam_attrs;
    uint16 total_num_interparams = 0;
    for (auto &&interparam : interparams) {
      auto &attr = interparam_attrs.emplace_back();
      detail::parseElseThrow(interparam, Grammer::SymArr, attr);
      uint16 array_size = 1;
      for (auto &&dim : attr.second) {
        array_size *= dim;
      }
      total_num_interparams += array_size;
    }
    for (auto &&[name, dims] : interparam_attrs) {
      std::vector<uint16> idxs(dims.size(), 0);
      detail::traverseArraySpace(
          name, dims, idxs, 0, [&](const std::string &id) {
            auto iter = detail::insertElseThrow(
                ret.SymbolTable, id,
                GinExt::interparam_array(ret.x, ret.NumIntPar,
                                         total_num_interparams));
            ret.NumIntPar++;
            ret.InterParams.emplace_back(iter, Gin::ex(0));
          });
    }
  }

  // parse intermediate parameter definitions
  for (auto &&[sym_it, attr] : ret.InterParams) {
    if (auto text_yml = config_yml["IntermediateParamDefs"][sym_it->first];
        text_yml) {
      detail::parseElseThrow(text_yml.as<std::string>(), Grammer::DiffFunc,
                             attr);
    }
  }

  // parse equations
  for (auto &&[sym_it, attr] : ret.Equations) {
    if (auto text_yml = config_yml["Equations"][sym_it->first + "'"];
        text_yml) {
      detail::parseElseThrow(text_yml.as<std::string>(), Grammer::DiffFunc,
                             attr);
    }
  }

  Grammer::SymbolTable = nullptr;

  return ret;
}

} // namespace Parser

} // namespace RKI