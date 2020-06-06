#pragma once

#include <ginac/ginac.h>

namespace RKI {

namespace Gin = GiNaC;

namespace GinExt {

DECLARE_FUNCTION_3P(unkwn_array);
DECLARE_FUNCTION_3P(diff_array);
DECLARE_FUNCTION_1P(cstm_step);

#define UNKWN_PARAMLIST                                                        \
  const Gin::ex &x, const Gin::ex &i, const Gin::ex &num_eq
#define DIFF_PARAMLIST const Gin::ex &x, const Gin::ex &i, const Gin::ex &num_eq
#define STEP_PARAMLIST const Gin::ex &n

#define REGISTER_FUNC(func_name, prefix)                                       \
  REGISTER_FUNCTION(func_name,                                                 \
                    eval_func(prefix##Eval)                                    \
                        .derivative_func(prefix##Diff)                         \
                        .print_func<Gin::print_dflt>(prefix##Print))

Gin::ex unkwnEval(UNKWN_PARAMLIST) { return unkwn_array(x, i, num_eq).hold(); }

Gin::ex unkwnDiff(UNKWN_PARAMLIST, unsigned o) {
  if (o == 0)
    return diff_array(x, i, num_eq).hold();
  else
    return 0;
}

void unkwnPrint(UNKWN_PARAMLIST, const Gin::print_context &ctx) {
  ctx.s << "y[" << i << "]";
}

Gin::ex diffEval(DIFF_PARAMLIST) { return diff_array(x, i, num_eq).hold(); }

Gin::ex diffDiff(DIFF_PARAMLIST, unsigned o) {
  if (o == 0)
    return diff_array(x, i + num_eq, num_eq).hold();
  else
    return 0;
}

void diffPrint(DIFF_PARAMLIST, const Gin::print_context &ctx) {
  ctx.s << "dy[" << i << "]";
}

Gin::ex stepEval(STEP_PARAMLIST) { return cstm_step(n).hold(); }

Gin::ex stepDiff(STEP_PARAMLIST, unsigned) { return 0; }

void stepPrint(STEP_PARAMLIST, const Gin::print_context &ctx) {
  ctx.s << "((" << n << ")>= 0.0)";
}

REGISTER_FUNC(unkwn_array, unkwn);
REGISTER_FUNC(diff_array, diff);
REGISTER_FUNC(cstm_step, step);

#undef UNKWN_PARAMLIST
#undef DIFF_PARAMLIST
#undef STEP_PARAMLIST
#undef REGISTER_FUNC

} // namespace GinExt

} // namespace RKI