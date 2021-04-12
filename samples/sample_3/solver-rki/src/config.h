#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

#define IN const
#define OUT
#define IN_OUT

// Configurable system constants
constexpr uint64_t NUM_SYS = 5;
constexpr uint16_t NUM_EQ = 1;
constexpr uint16_t NUM_PAR = 1;
constexpr uint16_t NUM_INTPAR = 0;
constexpr double X_BEGIN = 0;
constexpr double X_END = 100;
constexpr double STEP_SIZE = 0.01;

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
                            IN_OUT uint64_t f_cnts_global[NUM_SYS]);
