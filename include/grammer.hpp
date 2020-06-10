#pragma once

#include <boost/fusion/include/std_pair.hpp>
#include <boost/spirit/home/x3.hpp>
#include <ginac/ginac.h>

#include "ginac_ext.hpp"

namespace RKI {

namespace X3 = boost::spirit::x3;
namespace Fusion = boost::fusion;

namespace Grammer {

static const Gin::symtab *SymbolTable = nullptr;

// Rule declarations
static const X3::rule<class DiffFunc, Gin::ex> DiffFunc;
static const X3::rule<class F, Gin::ex> F;
static const X3::rule<class G, Gin::ex> G;
static const X3::rule<class H, Gin::ex> H;
static const X3::rule<class Id, std::string> Id;
static const X3::rule<class Sym, std::string> Sym;
static const X3::rule<class SymArr, std::pair<std::string, std::vector<uint16>>>
    SymArr;

#define VAL X3::_val(ctx)
#define ATTR X3::_attr(ctx)
#define ATTRAT(x) Fusion::at_c<x>(ATTR)

// Semantic actions
static const auto initAction = [](auto &ctx) { VAL = ATTR; };
static const auto plusAction = [](auto &ctx) { VAL += ATTR; };
static const auto minusAction = [](auto &ctx) { VAL -= ATTR; };
static const auto multiplyAction = [](auto &ctx) { VAL *= ATTR; };
static const auto divideAction = [](auto &ctx) { VAL /= ATTR; };
static const auto negateAction = [](auto &ctx) { VAL = -ATTR; };
static const auto sinAction = [](auto &ctx) { VAL = Gin::sin(ATTR); };
static const auto cosAction = [](auto &ctx) { VAL = Gin::cos(ATTR); };
static const auto tanAction = [](auto &ctx) { VAL = Gin::tan(ATTR); };
static const auto secAction = [](auto &ctx) {
  VAL = Gin::numeric(1) / Gin::cos(ATTR);
};
static const auto cscAction = [](auto &ctx) {
  VAL = Gin::numeric(1) / Gin::sin(ATTR);
};
static const auto cotAction = [](auto &ctx) {
  VAL = Gin::numeric(1) / Gin::tan(ATTR);
};
static const auto sinhAction = [](auto &ctx) { VAL = Gin::sinh(ATTR); };
static const auto coshAction = [](auto &ctx) { VAL = Gin::cosh(ATTR); };
static const auto tanhAction = [](auto &ctx) { VAL = Gin::tanh(ATTR); };
static const auto expAction = [](auto &ctx) { VAL = Gin::exp(ATTR); };
static const auto logAction = [](auto &ctx) { VAL = Gin::log(ATTR); };
static const auto powAction = [](auto &ctx) {
  VAL = Gin::pow(ATTRAT(0), ATTRAT(1));
};
static const auto stepAction = [](auto &ctx) { VAL = GinExt::cstm_step(ATTR); };
static const auto idAction = [](auto &ctx) {
  auto it = SymbolTable->find(ATTR);
  if (it == SymbolTable->end()) {
    throw std::runtime_error("Undeclared identifier detected: " + ATTR);
  }
  VAL = it->second;
};
static const auto numAction = [](auto &ctx) { VAL = Gin::numeric(ATTR); };

// Rule definitions
static const auto DiffFunc_def = F[initAction] >>
                                 *((X3::lit('+') > F)[plusAction] |
                                   (X3::lit('-') > F)[minusAction]);

static const auto F_def = G[initAction] >>
                          *((X3::lit('*') > G)[multiplyAction] |
                            (X3::lit('/') > G)[divideAction]);

static const auto G_def = (X3::lit('+') > H)[initAction] |
                          (X3::lit('-') > H)[negateAction] | H[initAction];

static const auto H_def =
    (X3::lit("sin(") > DiffFunc > X3::lit(')'))[sinAction] |
    (X3::lit("cos(") > DiffFunc > X3::lit(')'))[cosAction] |
    (X3::lit("tan(") > DiffFunc > X3::lit(')'))[tanAction] |
    (X3::lit("sec(") > DiffFunc > X3::lit(')'))[secAction] |
    (X3::lit("csc(") > DiffFunc > X3::lit(')'))[cscAction] |
    (X3::lit("cot(") > DiffFunc > X3::lit(')'))[cotAction] |
    (X3::lit("sinh(") > DiffFunc > X3::lit(')'))[sinhAction] |
    (X3::lit("tanh(") > DiffFunc > X3::lit(')'))[tanhAction] |
    (X3::lit("cosh(") > DiffFunc > X3::lit(')'))[coshAction] |
    (X3::lit("exp(") > DiffFunc > X3::lit(')'))[expAction] |
    (X3::lit("log(") > DiffFunc > X3::lit(')'))[logAction] |
    (X3::lit("pow(") > DiffFunc > X3::lit(',') > DiffFunc >
     X3::lit(')'))[powAction] |
    (X3::lit("step(") > DiffFunc > X3::lit(')'))[stepAction] |
    (X3::lit('(') > DiffFunc > X3::lit(')'))[initAction] | Id[idAction] |
    X3::long_[numAction] | X3::double_[numAction];

static const auto Id_def =
    X3::lexeme[X3::alpha >> *(X3::alnum | X3::char_('_')) >>
               *(X3::char_('[') > *X3::digit > X3::char_(']'))];

static const auto Sym_def =
    X3::lexeme[X3::alpha >> *(X3::alnum | X3::char_('_'))];

static const auto SymArr_def =
    X3::lexeme[Sym >> *(X3::lit('(') > X3::uint16 > X3::lit(')'))];

BOOST_SPIRIT_DEFINE(DiffFunc, F, G, H, Id, Sym, SymArr);

#undef VAL
#undef ATTR
#undef ATTRAT

} // namespace Grammer

} // namespace RKI