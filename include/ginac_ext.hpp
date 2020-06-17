#pragma once

#include <ginac/ginac.h>

namespace RKI {

namespace Gin = GiNaC;

namespace GinExt {

DECLARE_FUNCTION_3P(unknown_array);
DECLARE_FUNCTION_3P(diffunknown_array);
DECLARE_FUNCTION_1P(param_array);
DECLARE_FUNCTION_3P(interparam_array);
DECLARE_FUNCTION_1P(cstm_step);

#define UNKWN_PARAMLIST                                                        \
  const Gin::ex &x, const Gin::ex &i, const Gin::ex &num_eq
#define DIFF_PARAMLIST const Gin::ex &x, const Gin::ex &i, const Gin::ex &num_eq
#define PARAM_PARAMLIST const Gin::ex &i
#define INTERPARAM_PARAMLIST                                                   \
  const Gin::ex &x, const Gin::ex &i, const Gin::ex &num_intparam
#define STEP_PARAMLIST const Gin::ex &n

#define REGISTER_FUNC(func_name, prefix)                                       \
  REGISTER_FUNCTION(func_name,                                                 \
                    eval_func(prefix##Eval)                                    \
                        .derivative_func(prefix##Diff)                         \
                        .print_func<Gin::print_csrc>(prefix##Print))

Gin::ex unknownEval(UNKWN_PARAMLIST) {
  return unknown_array(x, i, num_eq).hold();
}

Gin::ex unknownDiff(UNKWN_PARAMLIST, unsigned o) {
  if (o == 0)
    return diffunknown_array(x, i, num_eq).hold();
  else
    return 0;
}

void unknownPrint(UNKWN_PARAMLIST, const Gin::print_context &ctx) {
  ctx.s << "y[" << Gin::to_long(Gin::ex_to<Gin::numeric>(i)) << "]";
}

Gin::ex diffunknownEval(DIFF_PARAMLIST) {
  return diffunknown_array(x, i, num_eq).hold();
}

Gin::ex diffunknownDiff(DIFF_PARAMLIST, unsigned o) {
  if (o == 0)
    return diffunknown_array(x, i + num_eq, num_eq).hold();
  else
    return 0;
}

void diffunknownPrint(DIFF_PARAMLIST, const Gin::print_context &ctx) {
  auto i_int = Gin::to_long(Gin::ex_to<Gin::numeric>(i));
  auto num_eq_int = Gin::to_long(Gin::ex_to<Gin::numeric>(num_eq));
  ctx.s << "dy[" << (i_int / num_eq_int) << "][" << (i_int % num_eq_int) << "]";
}

Gin::ex paramEval(PARAM_PARAMLIST) { return param_array(i).hold(); }

Gin::ex paramDiff(PARAM_PARAMLIST, unsigned) { return 0; }

void paramPrint(PARAM_PARAMLIST, const Gin::print_context &ctx) {
  ctx.s << "g[" << Gin::to_long(Gin::ex_to<Gin::numeric>(i)) << "]";
}

Gin::ex interparamEval(INTERPARAM_PARAMLIST) {
  return interparam_array(x, i, num_intparam).hold();
}

Gin::ex interparamDiff(INTERPARAM_PARAMLIST, unsigned o) {
  if (o == 0)
    return interparam_array(x, i + num_intparam, num_intparam);
  else
    return 0;
}

void interparamPrint(INTERPARAM_PARAMLIST, const Gin::print_context &ctx) {
  auto i_int = Gin::to_long(Gin::ex_to<Gin::numeric>(i));
  auto num_intparam_int = Gin::to_long(Gin::ex_to<Gin::numeric>(num_intparam));
  ctx.s << "ig[" << (i_int / num_intparam_int) << "]["
        << (i_int % num_intparam_int) << "]";
}

Gin::ex stepEval(STEP_PARAMLIST) { return cstm_step(n).hold(); }

Gin::ex stepDiff(STEP_PARAMLIST, unsigned) { return 0; }

void stepPrint(STEP_PARAMLIST, const Gin::print_context &ctx) {
  ctx.s << "((" << n << ") >= 0.0)";
}

REGISTER_FUNC(unknown_array, unknown);
REGISTER_FUNC(diffunknown_array, diffunknown);
REGISTER_FUNC(param_array, param);
REGISTER_FUNC(interparam_array, interparam);
REGISTER_FUNC(cstm_step, step);

#undef UNKWN_PARAMLIST
#undef DIFF_PARAMLIST
#undef PARAM_PARAMLIST
#undef INTERPARAM_PARAMLIST
#undef STEP_PARAMLIST
#undef REGISTER_FUNC

} // namespace GinExt

} // namespace RKI