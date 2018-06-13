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
/// \file Utils.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_GPU_UTILS_H_
#define TRACKINGITSU_INCLUDE_GPU_UTILS_H_

#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/gpu/Stream.h"

namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{

namespace Utils {

namespace Host {
dim3 getBlockSize(const int);
dim3 getBlockSize(const int, const int);
dim3 getBlockSize(const int, const int, const int);
dim3 getBlocksGrid(const dim3&, const int);
dim3 getBlocksGrid(const dim3&, const int, const int);

void gpuMalloc(void**, const int);
void gpuFree(void*);
void gpuMemset(void *, int, int);
void gpuMemcpyHostToDevice(void *, const void *, int);
void gpuMemcpyHostToDeviceAsync(void *, const void *, int, Stream&);
void gpuMemcpyDeviceToHost(void *, const void *, int);
void gpuStartProfiler();
void gpuStopProfiler();
}
#if TRACKINGITSU_OCL_MODE
		int findNearestDivisor(const int numToRound, const int divisor);
		int roundUp(const int numToRound, const int multiple);
		char *OCLErr_code (int err_in);
		compute::kernel CreateBoostKernelFromFile(compute::context boostContext , compute::device boostDevice, const char* fileName,const char* kernelName);
		cl::Kernel CreateKernelFromFile(cl::Context, cl::Device device, const char* fileName, const char* kernelName);
#endif



namespace Device {
GPU_DEVICE int getLaneIndex();
GPU_DEVICE int shareToWarp(const int, const int);
GPU_DEVICE int gpuAtomicAdd(int*, const int);
}
}

}
}
}
}

#endif /* TRACKINGITSU_INCLUDE_GPU_UTILS_H_ */
