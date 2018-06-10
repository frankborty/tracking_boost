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
/// \file Context.h
/// \brief
///

#ifndef TRAKINGITSU_INCLUDE_GPU_CONTEXT_H_
#define TRAKINGITSU_INCLUDE_GPU_CONTEXT_H_

#include <string>
#include <vector>
#include <iostream>
#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/Constants.h"

namespace o2 {
namespace ITS {
namespace CA {
namespace GPU {

struct DeviceProperties final {
#if TRACKINGITSU_CUDA_MODE
	std::string name;
	int gpuProcessors;
	int cudaCores;
	long globalMemorySize;
	long constantMemorySize;
	long sharedMemorySize;
	long maxClockRate;
	int busWidth;
	long l2CacheSize;
	long registersPerBlock;
	int warpSize;
	int maxThreadsPerBlock;
	int maxBlocksPerSM;
	dim3 maxThreadsDim;
	dim3 maxGridDim;
#else
	std::string name;
	long globalMemorySize;
	int warpSize;

	std::string vendor;
	std::size_t maxComputeUnits;
	std::size_t maxWorkGroupSize;
	std::size_t maxWorkItemDimension;

	//bost
	compute::device boostDevice;
	compute::context boostContext;
	compute::command_queue boostCommandQueue;
	compute::command_queue boostCommandQueues[Constants::ITS::LayersNumber];
	compute::kernel countTrackletsBoostKernel;
	compute::kernel computeTrackletsBoostKernel;
	compute::kernel countCellsBoostKernel;
	compute::kernel computeCellsBoostKernel;
	int fixedWorkSize = 32;

#endif
};

class Context final {
public:
	static Context& getInstance();

	Context(const Context&);
	Context& operator=(const Context&);

	const DeviceProperties& getDeviceProperties();
	const DeviceProperties& getBoostDeviceProperties();
	const DeviceProperties& getDeviceProperties(const int);

private:
	Context();
	~Context() = default;

#ifdef TRACKINGITSU_OCL_MODE
	int iCurrentDevice;
	int mDevicesNum;
	std::vector<DeviceProperties> mDeviceProperties;
	DeviceProperties mBoostDeviceProperties;
#else
	int mDevicesNum;
	std::vector<DeviceProperties> mDeviceProperties;
#endif
};

}
}
}
}

#endif /* TRAKINGITSU_INCLUDE_GPU_CONTEXT_H_ */
