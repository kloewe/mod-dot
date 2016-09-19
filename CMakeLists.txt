project(dot)

add_compile_options(-std=c99 -Wall -Wextra -Wno-unused-parameter -Wconversion -Wshadow -pedantic)

add_library(dot_avx src/dot_avx.c)
target_compile_options(dot_avx PRIVATE -mavx -funroll-loops)

add_library(dot_sse2 src/dot_sse2.c)
target_compile_options(dot_sse2 PRIVATE -msse2)

add_library(dot_naive src/dot_naive.c)

add_library(dot src/dot.c)
target_include_directories(dot PRIVATE ../cpuinfo/src)
target_link_libraries(dot dot_avx dot_sse2 dot_naive cpuinfo)
