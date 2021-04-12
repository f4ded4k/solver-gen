project('solver-gen', 'cpp', default_options : ['cpp_std=c++17'])

cmake = import('cmake')

boost = dependency('boost')
yamlcpp = dependency('yaml-cpp')
ginac = cmake.subproject('GiNaC').dependency('ginac')
cxxopts = dependency('cxxopts')


inc = include_directories('include')

src = files('src/main.cpp')

executable('solver-gen', src, include_directories : inc, dependencies : [boost, yamlcpp, ginac, cxxopts])
