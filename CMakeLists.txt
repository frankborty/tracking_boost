cmake_minimum_required(VERSION 3.2.0)
project(TRACKING-ITSU C CXX)

include(CheckIncludeFileCXX)

set(TRACKINGITSU_TARGET_DEVICE CPU CACHE STRING "Target device where code must be run. Options are: CPU (default), GPU_CUDA, OPEN_CL")
set_property(CACHE TRACKINGITSU_TARGET_DEVICE PROPERTY STRINGS CPU GPU_CUDA OPEN_CL)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG -g -O0")
set(CMAKE_CXX_FLAGS_PROFILE "-pg" CACHE STRING "Flags used by the C++ compiler during profiling builds.")
set(CMAKE_CXX_FLAGS_MEMORYBENCHMARK "-DMEMORY_BENCHMARK" CACHE STRING "Flags used by the C++ compiler during memory benchmark builds.")
set(CMAKE_CXX_FLAGS_TIMEBENCHMARK "-DTIME_BENCHMARK" CACHE STRING "Flags used by the C++ compiler during time benchmark builds.")

MARK_AS_ADVANCED(CMAKE_CXX_FLAGS_PROFILE)
MARK_AS_ADVANCED(CMAKE_CXX_FLAGS_MEMORYBENCHMARK)
MARK_AS_ADVANCED(CMAKE_CXX_FLAGS_TIMEBENCHMARK)

set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING
  "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Profile MemoryBenchmark.")

check_include_file_cxx(valgrind/callgrind.h HAVE_VALGRIND)

if(TRACKINGITSU_TARGET_DEVICE STREQUAL GPU_CUDA)
    message("-- Compiling for NVIDIA CUDA enabled devices")

    set(CMAKE_CXX_FLAGS_PROFILE "${CMAKE_CXX_FLAGS_PROFILE} -DGPU_PROFILING_MODE")

    add_definitions(-DTRACKINGITSU_CUDA_COMPILE)

    include_directories(${TRACKING-ITSU_SOURCE_DIR}/include/gpu)
endif(TRACKINGITSU_TARGET_DEVICE STREQUAL GPU_CUDA)

if(TRACKINGITSU_TARGET_DEVICE STREQUAL OPEN_CL)
	add_definitions(-DTRACKINGITSU_OPEN_CL_COMPILE )
	find_package( OpenCL REQUIRED )
	include_directories( ${OpenCl_INCLUDE_DIRS} ${TRACKING-ITSU_SOURCE_DIR}/include/ITSReconstruction/CA/openCl ${TRACKING-ITSU_SOURCE_DIR}/src)


	find_package(Boost REQUIRED)
	include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
	message(${Boost_INCLUDE_DIRS})

endif(TRACKINGITSU_TARGET_DEVICE STREQUAL OPEN_CL)
	
include_directories(${TRACKING-ITSU_SOURCE_DIR}/include)
add_subdirectory(src)

add_executable(tracking-itsu-main main.cpp)

if(TRACKINGITSU_TARGET_DEVICE STREQUAL OPEN_CL)
	target_link_libraries(tracking-itsu-main src ${OpenCL_LIBRARIES})
endif(TRACKINGITSU_TARGET_DEVICE STREQUAL OPEN_CL)

target_link_libraries(tracking-itsu-main src)
