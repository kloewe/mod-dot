project(dot)

if(CMAKE_VERSION VERSION_LESS "2.8.12")
    function(add_compile_options)
        add_definitions(${ARGN})
    endfunction(add_compile_options)
endif()

add_compile_options(-std=c99 -Wall -Wextra -Wno-unused-parameter -Wconversion -Wshadow -pedantic)

add_library(dot_avx src/dot_avx.c)
add_library(dot_sse2 src/dot_sse2.c)
add_library(dot_naive src/dot_naive.c)
add_library(dot src/dot.c)

target_link_libraries(dot dot_avx dot_sse2 dot_naive cpuinfo)

if(CMAKE_VERSION VERSION_LESS "2.8.12")
    set_target_properties(dot_avx PROPERTIES COMPILE_FLAGS "-mavx -funroll-loops")
    set_target_properties(dot_sse2 PROPERTIES COMPILE_FLAGS "-msse2")
    include_directories(../cpuinfo/src)
else()
    target_compile_options(dot_avx PRIVATE -mavx -funroll-loops)
    target_compile_options(dot_sse2 PRIVATE -msse2)
    target_include_directories(dot PRIVATE ../cpuinfo/src)
endif()