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
/// \file Context.cu
/// \brief
///

#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/gpu/Context.h"
#include "ITSReconstruction/CA/gpu/Utils.h"
#include "ITSReconstruction/CA/gpu/StructGPUPrimaryVertex.h"
#define AMD_WAVEFRONT 		0x4043
#define NVIDIA_WAVEFRONT 	0x4003


namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{

Context::Context()
{
	std::vector<cl::Platform> platformList;
	std::vector<cl::Device> deviceList;
	std::vector<std::size_t> sizeDim;
	std::string info;
	int scelta=0;
	std::vector<compute::device> boostDevicesList;

	try{

		boostDevicesList = compute::system::devices();
		for(int i=0;i<(int)boostDevicesList.size();i++)
			std::cout<<"["<<i<<"]: "<<boostDevicesList[i].name()<<std::endl;
		std::cout<<"Select device:";
		std::cin>>scelta;
		iCurrentDevice=scelta;
		mDevicesNum=boostDevicesList.size();
		mBoostDeviceProperties.boostDevice=boostDevicesList[scelta];
		mBoostDeviceProperties.boostContext=compute::context(mBoostDeviceProperties.boostDevice);
		mBoostDeviceProperties.boostCommandQueue=compute::command_queue(mBoostDeviceProperties.boostContext,mBoostDeviceProperties.boostDevice);

		for(int i=0;i<Constants::ITS::TrackletsPerRoad;i++)
			mBoostDeviceProperties.boostCommandQueues[i]=compute::command_queue(mBoostDeviceProperties.boostContext,mBoostDeviceProperties.boostDevice);

		char deviceVendor[255];
		int warpSize=0;
		clGetDeviceInfo(mBoostDeviceProperties.boostDevice.id(), CL_DEVICE_VENDOR, sizeof(deviceVendor), deviceVendor, NULL);
		if(strstr(deviceVendor,"NVIDIA")!=NULL || strstr(deviceVendor,"nvidia")!=NULL || strstr(deviceVendor,"Nvidia")!=NULL){
			//std::cout<<">> NVIDIA" << std::endl;
			clGetDeviceInfo(mBoostDeviceProperties.boostDevice.id(), NVIDIA_WAVEFRONT, sizeof(warpSize), &warpSize, NULL);
		}
		else if(strstr(deviceVendor,"AMD")!=NULL || strstr(deviceVendor,"amd")!=NULL || strstr(deviceVendor,"Amd")!=NULL){
			//std::cout<<">> NVIDIA" << std::endl;
			clGetDeviceInfo(mBoostDeviceProperties.boostDevice.id(), AMD_WAVEFRONT, sizeof(warpSize), &warpSize, NULL);
		}
		else{
			warpSize=128;
		}


		mBoostDeviceProperties.warpSize=warpSize;
		//mBoostDeviceProperties.countTrackletsBoostKernel=GPU::Utils::CreateBoostKernelFromFile(mBoostDeviceProperties.boostContext,mBoostDeviceProperties.boostDevice,"./src/kernel/countLayerTracklets.cl","countLayerTracklets");
		mBoostDeviceProperties.computeTrackletsBoostKernel=GPU::Utils::CreateBoostKernelFromFile(mBoostDeviceProperties.boostContext,mBoostDeviceProperties.boostDevice,"./src/kernel/computeLayerTracklets.cl","computeLayerTracklets");
		//mBoostDeviceProperties.countCellsBoostKernel=GPU::Utils::CreateBoostKernelFromFile(mBoostDeviceProperties.boostContext,mBoostDeviceProperties.boostDevice,"./src/kernel/countLayerCells.cl","countLayerCells");
		mBoostDeviceProperties.computeCellsBoostKernel=GPU::Utils::CreateBoostKernelFromFile(mBoostDeviceProperties.boostContext,mBoostDeviceProperties.boostDevice,"./src/kernel/computeLayerCells.cl","computeLayerCells");


	}catch(const cl::Error &err){
		std::string errString=Utils::OCLErr_code(err.err());
		//std::cout<< errString << std::endl;
		throw std::runtime_error { errString };
	}
}


Context& Context::getInstance()
{
  static Context gpuContext;
  return gpuContext;
}

const DeviceProperties& Context::getDeviceProperties()
{
  return getDeviceProperties(iCurrentDevice);
}

const DeviceProperties& Context::getBoostDeviceProperties()
{
  return mBoostDeviceProperties;
}

const DeviceProperties& Context::getDeviceProperties(const int deviceIndex)
{
	return mDeviceProperties[deviceIndex];

}

}
}
}
}
