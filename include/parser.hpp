#pragma once

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
  uint16 NumEq = 0, NumPar = 0, NumDiff = 2;
  double Stepsize = 0, xBegin = 0, xEnd = 0;

  Gin::symbol x;
  Gin::symtab SymbolTable;

  std::vector<decltype(SymbolTable)::iterator> Parameters;
  std::vector<std::pair<decltype(SymbolTable)::iterator, Gin::ex>> Equations;
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

template <typename Callable_, typename... Args>
void traverseAllIdxs(const std::vector<uint16> &dims, std::vector<uint16> &idxs,
                     uint16 idx, const Callable_ &callable) {
  if (idx == dims.size()) {
    callable(idxs);
    return;
  }
  for (uint16 i = 1; i <= dims[idx]; ++i) {
    idxs[idx] = i;
    traverseAllIdxs(dims, idxs, idx + 1, callable);
  }
}

} // namespace detail

ParsedObject parse(const fs::path &config_path) {
  ParsedObject ret;
  auto config_yml = YAML::LoadFile(config_path);

  // Parse Constants
  ret.NumSys = config_yml["Constants"]["NumSys"].as<decltype(ret.NumSys)>();
  ret.Stepsize =
      config_yml["Constants"]["Stepsize"].as<decltype(ret.Stepsize)>();
  ret.xBegin = config_yml["Constants"]["Begin"].as<decltype(ret.xBegin)>();
  ret.xEnd = config_yml["Constants"]["End"].as<decltype(ret.xEnd)>();

  // Parse IndependentVar
  std::string var = config_yml["Constants"]["IndependentVar"].as<std::string>();
  std::string attr;
  auto it = var.begin();
  auto succ = X3::phrase_parse(it, var.end(), Grammer::Sym, X3::space, attr);
  detail::throwIf(!succ || it != var.end(),
                  "Failed to parse independent variable");
  ret.x = Gin::symbol("x");
  ret.SymbolTable[attr] = ret.x;

  // Parse Parameters
  auto params = config_yml["Parameters"].as<std::vector<std::string>>();
  for (auto &&param : params) {
    std::pair<std::string, std::vector<uint16>> attr;
    auto it = param.begin();
    auto succ =
        X3::phrase_parse(it, param.end(), Grammer::SymArr, X3::space, attr);
    detail::throwIf(!succ || it != param.end(),
                    "Failed to parse parameter: " + param);
    std::vector<uint16> idxs(attr.second.size(), 0);
    detail::traverseAllIdxs(
        attr.second, idxs, 0, [&](const std::vector<uint16> &idxs) {
          std::stringstream ss;
          ss << attr.first;
          for (auto &&idx : idxs) {
            ss << '[' << idx << ']';
          }
          auto iter = detail::insertElseThrow(
              ret.SymbolTable, ss.str(),
              Gin::symbol("g[" + std::to_string(ret.NumPar) + "]"));
          ret.Parameters.emplace_back(std::move(iter));
          ret.NumPar++;
        });
  }

  // Parse Unknowns
  auto eqs = [&]() {
    auto eqs_omap = config_yml["Equations"]
                        .as<std::vector<std::map<std::string, std::string>>>();
    std::vector<std::pair<std::string, std::string>> ret;
    for (auto &&omap : eqs_omap) {
      detail::throwIf(omap.size() != 1, "Invalid equation format");
      ret.emplace_back(omap.begin()->first, std::move(omap.begin()->second));
    }
    return ret;
  }();
  for (auto &&[unk, ignored] : eqs) {
    detail::throwIf(unk.empty() || unk.back() != '\'',
                    "Invalid unknown format");
    unk.pop_back();
    std::string attr;
    auto it = unk.begin();
    auto succ = X3::phrase_parse(it, unk.end(), Grammer::Sym, X3::space, attr);
    detail::throwIf(!succ || it != unk.end(),
                    "Failed to parse unknown: " + unk);
    auto iter = detail::insertElseThrow(
        ret.SymbolTable, attr,
        GinExt::unkwn_array(ret.x, ret.NumEq, eqs.size()));
    ret.Equations.emplace_back(std::move(iter), 0);
    ret.NumEq++;
  }

  // Parse Equations
  Grammer::SymbolTable = &ret.SymbolTable;
  for (uint16 i = 0; i < eqs.size(); ++i) {
    auto it = eqs[i].second.begin();
    auto succ = X3::phrase_parse(it, eqs[i].second.end(), Grammer::DiffFunc,
                                 X3::space, ret.Equations[i].second);
    detail::throwIf(!succ || it != eqs[i].second.end(),
                    "Failed to parse equation #" + std::to_string(i + 1));
  }
  Grammer::SymbolTable = nullptr;

  return ret;
}

} // namespace Parser

} // namespace RKI