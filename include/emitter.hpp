#include <filesystem>
#include <fstream>

#include "parser.hpp"

namespace RKI {

namespace fs = std::filesystem;

namespace Emitter {

struct ConfigDir {
  fs::path root, input, output, src, config;
};

ConfigDir configureDirectory(const fs::path &config_path) {
  ConfigDir ret;

  if (!fs::is_regular_file(config_path)) {
    throw std::runtime_error("Config file doesn't exist");
  }
  ret.root = config_path.parent_path();
  ret.input = ret.root / "input";
  ret.output = ret.root / "output";
  ret.src = ret.root / "src";
  ret.config = config_path;
  fs::create_directory(ret.input);
  fs::create_directory(ret.output);
  fs::create_directory(ret.src);

  return ret;
}

void emitSrcs(const ConfigDir &dir, const Parser::ParsedObject &obj) {
  std::ofstream file(dir.src / "config.h");

  std::string format_str = R"CPP(#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

#define IN const
#define OUT
#define IN_OUT

// Configurable system constants
constexpr uint64_t NUM_SYS = %1%;
constexpr uint16_t NUM_EQ = %2%;
constexpr uint16_t NUM_PAR = %3%;
constexpr double X_BEGIN = %4%;
constexpr double X_END = %5%;
constexpr double STEP_SIZE = %6%;

// Configurable solver constants
constexpr uint16_t NUM_DIFF = 2;
constexpr uint16_t NUM_STAGE = 3;
constexpr uint16_t PRIMARY_ORDER = 5;
constexpr uint16_t EMBEDDED_ORDER = 4;
constexpr double TINY = 1.0e-30;
constexpr double EPS = 1.0e-10;
constexpr double SAFETY = 0.9;
constexpr double P1 = 0.1;
constexpr double P2 = 5.0;

// Deduced compile-time constants
constexpr uint64_t NUM_POINT = (X_END - X_BEGIN) / STEP_SIZE + 1;

// Deduced solver constants
constexpr uint16_t DY_SIZE = NUM_EQ * NUM_DIFF;
constexpr uint64_t Y_GLOBAL_SIZE = NUM_SYS * NUM_EQ;
constexpr uint64_t G_GLOBAL_SIZE = NUM_SYS * NUM_PAR;
constexpr double PSHRINK = -1.0 / EMBEDDED_ORDER;
constexpr double PGROW = -1.0 / PRIMARY_ORDER;

__global__ void solver_main(IN double x,
                            IN_OUT double y_global[NUM_SYS * NUM_EQ],
                            IN double g_global[NUM_SYS * NUM_PAR],
                            IN_OUT uint64_t *eval_count);)CPP";

  file << (boost::format(format_str) % obj.NumSys % obj.NumEq % obj.NumPar %
           obj.xBegin % obj.xEnd % obj.Stepsize)
              .str();
  file.close();
  file.open(dir.src / "host.cu");
  format_str = R"CPP(#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "config.h"

using namespace std::chrono;

#define CUDA_ERROR(cuda_expr)                                                  \
  { cuda_error_check((cuda_expr), __FILE__, __LINE__); }

void cuda_error_check(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    cout << "CUDA Error: " << cudaGetErrorString(code) << ' ' << file << ' '
         << line << '\n';
    if (abort) {
      exit(code);
    }
  }
}

int main() {
  ios_base::sync_with_stdio(false);

  auto g = new double[G_GLOBAL_SIZE];
  auto y = new double[Y_GLOBAL_SIZE * NUM_POINT];

  fstream file("inputs/parameters.csv", ios::in);
  string line;
  char comma;
  for (uint64_t i = 0; i < NUM_SYS; ++i) {
    getline(file, line);
    stringstream ss(line);
    for (uint16_t j = 0; j < NUM_PAR; ++j) {
      ss >> g[i + NUM_SYS * j];
      ss >> comma;
    }
  }
  file.close();

  file.open("inputs/initial_values.csv", ios::in);
  for (uint64_t i = 0; i < NUM_SYS; ++i) {
    getline(file, line);
    stringstream ss(line);
    for (uint16_t j = 0; j < NUM_EQ; ++j) {
      ss >> y[i + NUM_SYS * j];
      ss >> comma;
    }
  }
  file.close();

  unsigned block_size;
  if (NUM_SYS < 4194304) {
    block_size = 64u;
  } else if (NUM_SYS < 8388608) {
    block_size = 128u;
  } else if (NUM_SYS < 16777216) {
    block_size = 256u;
  } else {
    block_size = 512u;
  }

  dim3 block_dim(block_size);
  dim3 grid_dim((NUM_SYS / block_size) + 1u);

  auto gpu_start_time = steady_clock::now();

  double *y_device, *g_device;

  CUDA_ERROR(cudaMalloc(&y_device, Y_GLOBAL_SIZE * sizeof(double)));
  CUDA_ERROR(cudaMalloc(&g_device, G_GLOBAL_SIZE * sizeof(double)));

  CUDA_ERROR(cudaMemcpy(y_device, y, Y_GLOBAL_SIZE * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_ERROR(cudaMemcpy(g_device, g, G_GLOBAL_SIZE * sizeof(double),
                        cudaMemcpyHostToDevice));

  uint64_t eval_count = 0, *eval_count_device;
  CUDA_ERROR(cudaMalloc(&eval_count_device, sizeof(uint64_t)));
  CUDA_ERROR(cudaMemcpy(eval_count_device, &eval_count, sizeof(uint64_t),
                        cudaMemcpyHostToDevice));

  double x_curr = X_BEGIN;

  for (uint64_t i = 1; i < NUM_POINT; ++i, x_curr += STEP_SIZE) {
    solver_main<<<grid_dim, block_dim>>>(x_curr, y_device, g_device,
                                         eval_count_device);

    CUDA_ERROR(cudaGetLastError());

    CUDA_ERROR(cudaMemcpy(y + i * NUM_SYS * NUM_EQ, y_device,
                          NUM_SYS * NUM_EQ * sizeof(double),
                          cudaMemcpyDeviceToHost));
  }

  CUDA_ERROR(cudaMemcpy(&eval_count, eval_count_device, sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));

  cudaFree(y_device);
  cudaFree(g_device);
  cudaFree(eval_count_device);

  auto gpu_finish_time = steady_clock::now();
  cout << "Computation ran for : "
       << duration_cast<milliseconds>(gpu_finish_time - gpu_start_time).count()
       << "ms\n";
  cout << "Number of function evaluations : " << eval_count << '\n';

  auto result_start_time = steady_clock::now();

  file.open("outputs/result.csv", ios::out);
  file << scientific << setprecision(8);
  decltype(Y_GLOBAL_SIZE * NUM_POINT) index = 0;
  x_curr = X_BEGIN;
  for (uint64_t i = 0; i < NUM_POINT; ++i, x_curr += STEP_SIZE) {
    file << x_curr << ',';
    for (uint64_t j = 0; j < NUM_EQ * NUM_SYS; ++j, (file << ',')) {
      file << y[index++];
    }
    file << '\n';
  }
  file.close();

  delete[] y;
  delete[] g;

  auto result_finish_time = steady_clock::now();
  cout << "Finished writing output to outputs/result.csv in "
       << duration_cast<milliseconds>(result_finish_time - result_start_time)
              .count()
       << "ms\n";
})CPP";

  file << format_str;
  file.close();
  file.open(dir.src / "device.cu");

  std::vector<Gin::ex> equations;
  for (auto &&[it, expr] : obj.Equations) {
    equations.emplace_back(it->second.diff(obj.x) == expr);
  }
  std::stringstream eqs_ss;
  for (int i = 0; i < obj.NumDiff; ++i) {
    for (auto &&eq : equations) {
      eqs_ss << "  " << eq.lhs() << " = " << eq.rhs() << ";\n";
      eq = eq.diff(obj.x);
    }
    eqs_ss << '\n';
  }

  format_str = R"CPP(#include "config.h"

__device__ void compute_system(IN double h, IN double x, IN double y[NUM_EQ],
                               IN double g[NUM_PAR], OUT double dy[DY_SIZE]) {
%1%

  double hx = h;
  for (uint16_t diff = 0; diff < NUM_DIFF; ++diff) {
    for (uint16_t i = 0; i < NUM_EQ; ++i) {
      dy[NUM_EQ * diff + i] *= hx;
    }
    hx *= h;
  }
}

__global__ void solver_main(IN double x, IN_OUT double y_global[Y_GLOBAL_SIZE],
                            IN double g_global[G_GLOBAL_SIZE],
                            IN_OUT uint64_t *eval_count) {
  const uint64_t sys_index =
      (uint64_t)blockDim.x * blockIdx.x + (uint64_t)threadIdx.x;

  if (sys_index < NUM_SYS) {
    double y[NUM_EQ], g[NUM_PAR];

    for (uint16_t i = 0; i < NUM_EQ; ++i) {
      y[i] = y_global[sys_index + NUM_SYS * i];
    }
    for (uint16_t i = 0; i < NUM_PAR; ++i) {
      g[i] = g_global[sys_index + NUM_SYS * i];
    }

    constexpr double a[NUM_DIFF][NUM_STAGE][NUM_STAGE] = {
        {{3.0 / 11.0}, {18.0 / 25.0, 0.0}},
        {{9.0 / 242.0}, {-9.0 / 15625.0, 4059.0 / 15625.0}}};

    constexpr double b[NUM_DIFF][NUM_STAGE] = {
        {1.0, 0.0, 0.0}, {53.0 / 648.0, 1331.0 / 4428.0, 3125.0 / 26568.0}};

    constexpr double eb[NUM_DIFF][NUM_STAGE] = {
        {783089.0 / 1417500.0, 3115871.0 / 9686250.0, 11705.0 / 92988.0},
        {5989.0 / 157500.0, 28919.0 / 157500.0, 1.0 / 10.0}};

    constexpr double c[NUM_STAGE] = {0.0, 3.0 / 11.0, 18.0 / 25.0};

    double x_curr = x, x_end = x + STEP_SIZE;
    double h = 0.5 * STEP_SIZE;

    while (x_curr < x_end) {
      (*eval_count) += 3;

      double y_temp[NUM_EQ];

      double k1[DY_SIZE];
      compute_system(h, x_curr + c[0], y, g, k1);

      for (uint16_t i = 0; i < NUM_EQ; ++i) {
        y_temp[i] = y[i] + a[0][0][0] * k1[i] + a[1][0][0] * k1[NUM_EQ + i];
      }

      double k2[DY_SIZE];
      compute_system(h, x_curr + c[1] * h, y_temp, g, k2);

      for (uint16_t i = 0; i < NUM_EQ; ++i) {
        y_temp[i] = y[i] + a[0][1][0] * k1[i] + a[0][1][1] * k2[i] +
                    a[1][1][0] * k1[NUM_EQ + i] + a[1][1][1] * k2[NUM_EQ + i];
      }

      double k3[DY_SIZE];
      compute_system(h, x_curr + c[2] * h, y_temp, g, k3);

      for (uint16_t i = 0; i < NUM_EQ; ++i) {
        y_temp[i] = y[i] + b[0][0] * k1[i] + b[0][1] * k2[i] + b[0][2] * k3[i] +
                    b[1][0] * k1[NUM_EQ + i] + b[1][1] * k2[NUM_EQ + i] +
                    b[1][2] * k3[NUM_EQ + i];
      }

      double err = 0.0;
      bool is_nan = false;
      for (uint16_t i = 0; i < NUM_EQ; ++i) {
        double err_curr =
            y_temp[i] - (y[i] + eb[0][0] * k1[i] + eb[0][1] * k2[i] +
                         eb[0][2] * k3[i] + eb[1][0] * k1[NUM_EQ + i] +
                         eb[1][1] * k2[NUM_EQ + i] + eb[1][2] * k3[NUM_EQ + i]);

        err = fmax(err, fabs(err_curr / (fabs(y[i]) + fabs(k1[i]) + TINY)));
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

    for (uint16_t i = 0; i < NUM_EQ; ++i) {
      y_global[sys_index + NUM_SYS * i] = y[i];
    }
  }
})CPP";

  file << (boost::format(format_str) % eqs_ss.str()).str();
  file.close();
}

} // namespace Emitter

} // namespace RKI