#include "config.h"

__device__ void compute_system(IN double h, IN double x, IN double y[NUM_EQ],
                               IN double g[NUM_PAR],
                               OUT double dy[NUM_DIFF][NUM_EQ]) {
  dy[0][0] = y[1];
  dy[0][1] = pow(y[1], 2.0) + -3.0 * 1.0 / (pow(y[0], 2.0) + g[0]);
  dy[1][0] = dy[0][1];
  dy[1][1] = 2.0 * dy[0][1] * y[1] +
             6.0 * 1.0 / pow(pow(y[0], 2.0) + g[0], 2.0) * dy[0][0] * y[0];

  double hx = h;
  for (uint16_t i = 0; i < NUM_DIFF; ++i) {
    for (uint16_t j = 0; j < NUM_EQ; ++j) {
      dy[i][j] *= hx;
    }
    hx *= h;
  }
}

__global__ void solver_main(IN double x, IN_OUT double y_global[Y_GLOBAL_SIZE],
                            IN double g_global[G_GLOBAL_SIZE],
                            IN_OUT uint64_t f_cnts_global[NUM_SYS]) {
  const uint64_t sys_index =
      (uint64_t)blockDim.x * blockIdx.x + (uint64_t)threadIdx.x;

  if (sys_index < NUM_SYS) {
    uint64_t f_cnt = 0;
    double y[NUM_EQ], g[NUM_PAR];

    for (uint16_t i = 0; i < NUM_EQ; ++i) {
      y[i] = y_global[sys_index + NUM_SYS * i];
    }
    for (uint16_t i = 0; i < NUM_PAR; ++i) {
      g[i] = g_global[sys_index + NUM_SYS * i];
    }

    constexpr double a[NUM_STAGE][NUM_STAGE][NUM_DIFF] = {
        {},
        {{(3.0 / 11.0), (9.0 / 242.0)}},
        {{(18.0 / 25.0), -(9.0 / 15625.0)}, {0.0, (4059.0 / 15625.0)}}};

    constexpr double b[NUM_STAGE][NUM_DIFF] = {{1.0, (53.0 / 648.0)},
                                               {0.0, (1331.0 / 4428.0)},
                                               {0.0, (3125.0 / 26568.0)}};

    constexpr double eb[NUM_STAGE][NUM_DIFF] = {
        {(783089.0 / 1417500.0), (5989.0 / 157500.0)},
        {(3115871.0 / 9686250.0), (28919.0 / 157500.0)},
        {(11705.0 / 92988.0), (1.0 / 10.0)}};

    constexpr double c[NUM_STAGE] = {0.0, (3.0 / 11.0), (18.0 / 25.0)};

    double x_curr = x, x_end = x + STEP_SIZE;
    double h = 0.5 * STEP_SIZE;

    while (x_curr < x_end) {
      f_cnt += NUM_STAGE;
      double y_temp[NUM_EQ];

      double k[NUM_STAGE][NUM_DIFF][NUM_EQ];
      compute_system(h, x_curr + h * c[0], y, g, k[0]);

      for (uint16_t stage = 1; stage < NUM_STAGE; ++stage) {
        for (uint16_t eq = 0; eq < NUM_EQ; ++eq) {
          y_temp[eq] = y[eq];
          for (uint16_t i = 0; i < stage; ++i) {
            for (uint16_t j = 0; j < NUM_DIFF; ++j) {
              y_temp[eq] += a[stage][i][j] * k[i][j][eq];
            }
          }
        }
        compute_system(h, x_curr + h * c[stage], y_temp, g, k[stage]);
      }

      for (uint16_t eq = 0; eq < NUM_EQ; ++eq) {
        y_temp[eq] = y[eq];
        for (uint16_t i = 0; i < NUM_STAGE; ++i) {
          for (uint16_t j = 0; j < NUM_DIFF; ++j) {
            y_temp[eq] += b[i][j] * k[i][j][eq];
          }
        }
      }

      double err = 0.0;
      bool is_nan = false;
      for (uint16_t eq = 0; eq < NUM_EQ; ++eq) {
        double ey = y[eq];
        for (uint16_t i = 0; i < NUM_STAGE; ++i) {
          for (uint16_t j = 0; j < NUM_DIFF; ++j) {
            ey += eb[i][j] * k[i][j][eq];
          }
        }
        double err_curr = y_temp[eq] - ey;

        err = fmax(err,
                   fabs(err_curr / (fabs(y[eq]) + fabs(k[0][0][eq]) + TINY)));
        is_nan |= isnan(err) | isnan(err_curr);
      }
      err /= EPS;

      if (is_nan) {
        h *= P1;
      } else if (err > 1.0) {
        h *= fmax(SAFETY * pow(err, PSHRINK), P1);
      } else {
        x_curr += h;
        for (uint16_t i = 0; i < NUM_EQ; ++i) {
          y[i] = y_temp[i];
        }
        h *= fmin(SAFETY * pow(err, PGROW), P2);
      }

      h = fmin(h, x_end - x_curr);
    }

    f_cnts_global[sys_index] += f_cnt;
    for (uint16_t i = 0; i < NUM_EQ; ++i) {
      y_global[sys_index + NUM_SYS * i] = y[i];
    }
  }
}