// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file Definitions.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_CADEFINITIONS_H_
#define TRACKINGITSU_INCLUDE_CADEFINITIONS_H_

#ifdef __OPENCL_C_VERSION__
#define CONSTEXPR __constant
#define CONST
#define INITINT
#define INITFLOAT
#define INITFLOAT3
#define FLOAT3 Float3Struct
typedef struct{
		int x;
		int y;
		int z;
		int w;
	}Int4Struct;

typedef struct{
		float x;
		float y;
		float z;
	}Float3Struct;

typedef struct{
		float x;
		float y;
	}Float2Struct;

#else
#define CONSTEXPR constexpr
#define CONST const
#define INITINT 	=-1
#define INITFLOAT 	=0.f
#define INITFLOAT3  ={0.f,0.f,0.f}
#define FLOAT3 float3
#include <array>

#if defined(TRACKINGITSU_CUDA_COMPILE) || defined (TRACKINGITSU_OPEN_CL_COMPILE)
# define TRACKINGITSU_GPU_MODE true
#else
# define TRACKINGITSU_GPU_MODE false
#endif

#if defined(TRACKINGITSU_OPEN_CL_COMPILE)
#define __CL_ENABLE_EXCEPTIONS //enable exceptions
#if defined(__APPLE__)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif
#ifndef USE_BOOST
	#define USE_BOOST
	#include "boost/compute.hpp"
	#include "boost/compute/types/complex.hpp"
	namespace compute = boost::compute;
#endif
# define TRACKINGITSU_OCL_MODE true
# define TRACKINGITSU_CUDA_MODE false
#elif(TRACKINGITSU_CUDA_COMPILE)
# define TRACKINGITSU_OCL_MODE false
# define TRACKINGITSU_CUDA_MODE true
#endif


#if defined(__CUDACC__)
# define TRACKINGITSU_GPU_COMPILING
#endif

#if defined(__CUDA_ARCH__)
# define TRACKINGITSU_GPU_DEVICE
#endif

#if defined(__CUDACC__)

# define GPU_HOST __host__
# define GPU_DEVICE __device__
# define GPU_HOST_DEVICE __host__ __device__
# define GPU_GLOBAL __global__
# define GPU_SHARED __shared__
# define GPU_SYNC __syncthreads()

# define MATH_ABS abs
# define MATH_ATAN2 atan2
# define MATH_MAX max
# define MATH_MIN min
# define MATH_SQRT sqrt

# include "ITSReconstruction/CA/gpu/Array.h"

template<typename T, std::size_t Size>
using GPUArray = o2::ITS::CA::GPU::Array<T, Size>;

typedef cudaStream_t GPUStream;

#else

# define GPU_HOST
# define GPU_DEVICE
# define GPU_HOST_DEVICE
# define GPU_GLOBAL
# define GPU_SHARED
# define GPU_SYNC

# define MATH_ABS std::abs
# define MATH_ATAN2 std::atan2
# define MATH_MAX std::max
# define MATH_MIN std::min
# define MATH_SQRT std::sqrt

typedef struct _dim3 { unsigned int x, y, z; } dim3;
typedef struct _int4 { int x, y, z, w; } int4;
typedef struct _float2 { float x, y; } float2;
typedef struct _float3 { float x, y, z; } float3;
typedef struct _float4 { float x, y, z, w; } float4;

template<typename T, std::size_t Size>
using GPUArray = std::array<T, Size>;

typedef struct _dummyStream {} GPUStream;

#endif

#endif

#endif /* TRACKINGITSU_INCLUDE_CADEFINITIONS_H_ */
