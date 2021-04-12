#include <chrono>
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
  auto g = new double[G_GLOBAL_SIZE];
  auto y = new double[Y_GLOBAL_SIZE * NUM_POINT];
  auto pts = new double[NUM_POINT];
  auto f_cnts = new uint64_t[NUM_SYS];

  ifstream ifile("input/parameters.csv");
  string line;
  char comma;
  for (uint64_t i = 0; i < NUM_SYS; ++i) {
    if (getline(ifile, line)) {
      stringstream ss(line);
      for (uint16_t j = 0; j < NUM_PAR; ++j) {
        if (j != 0)
          ss >> comma;
        ss >> g[i + NUM_SYS * j];
        if (ss.fail()) {
          std::cerr << "Failed to parse parameters.csv line : " << i + 1
                    << '\n';
          return 1;
        }
      }
    } else {
      std::cerr << "Failed to parse parameters.csv line : " << i + 1 << '\n';
      return 1;
    }
  }

  ifile.close();
  ifile.open("input/initial_values.csv");
  for (uint64_t i = 0; i < NUM_SYS; ++i) {
    if (getline(ifile, line)) {
      stringstream ss(line);
      for (uint16_t j = 0; j < NUM_EQ; ++j) {
        if (j != 0)
          ss >> comma;
        ss >> y[i + NUM_SYS * j];
        if (ss.fail()) {
          std::cerr << "Failed to parse initial_values.csv line : " << i + 1
                    << '\n';
          return 1;
        }
      }
    } else {
      std::cerr << "Failed to parse initial_values.csv line : " << i + 1
                << '\n';
      return 1;
    }
  }

  ifile.close();

  for (uint64_t i = 0; i < NUM_SYS; ++i) {
    f_cnts[i] = 0;
  }

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
  uint64_t *f_cnts_device;

  CUDA_ERROR(cudaMalloc(&y_device, Y_GLOBAL_SIZE * sizeof(double)));
  CUDA_ERROR(cudaMalloc(&g_device, G_GLOBAL_SIZE * sizeof(double)));
  CUDA_ERROR(cudaMalloc(&f_cnts_device, NUM_SYS * sizeof(uint64_t)));

  CUDA_ERROR(cudaMemcpy(y_device, y, Y_GLOBAL_SIZE * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_ERROR(cudaMemcpy(g_device, g, G_GLOBAL_SIZE * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_ERROR(cudaMemcpy(f_cnts_device, f_cnts, NUM_SYS * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));

  pts[0] = X_BEGIN;
  for (uint64_t i = 1; i < NUM_POINT; ++i) {
    pts[i] = pts[i - 1] + STEP_SIZE;
  }

  for (uint64_t i = 1; i < NUM_POINT; ++i) {
    solver_main<<<grid_dim, block_dim>>>(pts[i - 1], y_device, g_device,
                                         f_cnts_device);

    CUDA_ERROR(cudaGetLastError());

    CUDA_ERROR(cudaMemcpy(y + i * NUM_SYS * NUM_EQ, y_device,
                          NUM_SYS * NUM_EQ * sizeof(double),
                          cudaMemcpyDeviceToHost));
  }

  CUDA_ERROR(cudaMemcpy(f_cnts, f_cnts_device, NUM_SYS * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));

  cudaFree(y_device);
  cudaFree(g_device);
  cudaFree(f_cnts_device);

  auto gpu_finish_time = steady_clock::now();
  cout << "Solver ran for : "
       << duration_cast<milliseconds>(gpu_finish_time - gpu_start_time).count()
       << "ms\n";

  auto result_start_time = steady_clock::now();

  ofstream ofile;
  for (uint64_t i = 0; i < NUM_SYS; ++i) {
    ofile.open("output/result_" + to_string(i + 1) + ".csv");
    ofile << "Number of evaluations : " << f_cnts[i] << '\n';
    ofile << scientific << setprecision(10);
    for (uint64_t j = 0; j < NUM_POINT; ++j) {
      ofile << pts[j];
      for (uint16_t k = 0; k < NUM_EQ; ++k) {
        ofile << ", " << y[j * NUM_SYS * NUM_EQ + k * NUM_SYS + i];
      }
      ofile << '\n';
    }
    ofile.close();
  }

  delete[] y;
  delete[] g;
  delete[] pts;
  delete[] f_cnts;

  auto result_finish_time = steady_clock::now();
  cout << "Finished writing results in "
       << duration_cast<milliseconds>(result_finish_time - result_start_time)
              .count()
       << "ms\n";
}