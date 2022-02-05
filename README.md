# solver-gen

Given a system of parametric ordinary differential equations combined with set of initial value & parameter sets, emits CUDA + C++ sources and executable to solve the ODE system for each initial value & parameter set in parallel on CUDA-capable devices.
It also supports generation of symbolic derivates of the ODE system required for the extended Runge-Kutta method as designed during internship at Centre for Integrative Biology and Systems medicinE (IBSE), IIT Madras.
