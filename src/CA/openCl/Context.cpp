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
	std::size_t iPlatformList;
	std::size_t iTotalDevice=0;
	int scelta=0;
	std::vector<compute::device> boostDevicesList;


	  clock_t t1, t2;
	  float totalTime = 0.f;
	try{

//boost
	std::cout<<"boost-start"<<std::endl;
	t1 = clock();



	boostDevicesList = compute::system::devices();
	for(int i=0;i<(int)boostDevicesList.size();i++)
		std::cout<<"["<<i<<"]: "<<boostDevicesList[i].name()<<std::endl;
	std::cout<<"Select device:";
	std::cin>>scelta;

	mBoostDeviceProperties.boostDevice=boostDevicesList[scelta];
	mBoostDeviceProperties.boostContext=compute::context(mBoostDeviceProperties.boostDevice);
	mBoostDeviceProperties.boostCommandQueue=compute::command_queue(mBoostDeviceProperties.boostContext,mBoostDeviceProperties.boostDevice);

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
	mBoostDeviceProperties.countTrackletsBoostKernel=GPU::Utils::CreateBoostKernelFromFile(mBoostDeviceProperties.boostContext,mBoostDeviceProperties.boostDevice,"./src/kernel/countLayerTracklets.cl","countLayerTracklets");
	mBoostDeviceProperties.computeTrackletsBoostKernel=GPU::Utils::CreateBoostKernelFromFile(mBoostDeviceProperties.boostContext,mBoostDeviceProperties.boostDevice,"./src/kernel/computeLayerTracklets.cl","computeLayerTracklets");
	mBoostDeviceProperties.countCellsBoostKernel=GPU::Utils::CreateBoostKernelFromFile(mBoostDeviceProperties.boostContext,mBoostDeviceProperties.boostDevice,"./src/kernel/countLayerCells.cl","countLayerCells");
	mBoostDeviceProperties.computeCellsBoostKernel=GPU::Utils::CreateBoostKernelFromFile(mBoostDeviceProperties.boostContext,mBoostDeviceProperties.boostDevice,"./src/kernel/computeLayerCells.cl","computeLayerCells");




	t2 = clock();
	totalTime = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);

	std::cout<<"boost-endClock: "<<totalTime<<std::endl;
///////

scelta=0;
t1 = clock();

std::cout<<"ocl-start"<<std::endl;
		// Get the list of platform
		cl::Platform::get(&platformList);
		iPlatformList=platformList.size();
		// Pick first platform

		//std::cout << "There are " << iPlatformList << " platform" << std::endl;
		//std::cout << std::endl;
		for(int iPlatForm=0;iPlatForm<(int)iPlatformList;iPlatForm++){
			//std::cout << "Platform #" << iPlatForm+1 << std::endl;
			cl::Context context;
			try{
				cl_context_properties cprops[] = {
				CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[iPlatForm])(), 0};
				context=cl::Context(CL_DEVICE_TYPE_ALL, cprops);
			}
			catch(const cl::Error &err){
					std::string errString=Utils::OCLErr_code(err.err());
					std::cout<<"No device found for platform #"<<iPlatForm<< std::endl;
					continue;
			}


			//print platform information
			platformList[iPlatForm].getInfo(CL_PLATFORM_NAME,&info);
			//std::cout << "Name:" 	<< info << std::endl;
			platformList[iPlatForm].getInfo(CL_PLATFORM_VENDOR,&info);
			//std::cout << "Vendor:"	<< info << std::endl;
			platformList[iPlatForm].getInfo(CL_PLATFORM_VERSION,&info);
			//std::cout << "Version: "<< info << std::endl;


			// Get devices associated with the first platform
			platformList[iPlatForm].getDevices(CL_DEVICE_TYPE_ALL,&deviceList);
			mDevicesNum=deviceList.size();
			mDeviceProperties.resize(iTotalDevice+mDevicesNum, DeviceProperties { });

			//std::cout << "There are " << mDevicesNum << " devices" << std::endl;

			for(int iDevice=0;iDevice<mDevicesNum;iDevice++){

				deviceList[iDevice].getInfo(CL_DEVICE_VENDOR,&(mDeviceProperties[iTotalDevice].vendor));
				//std::cout << "		Device vendor: " << mDeviceProperties[iTotalDevice].vendor << std::endl;
				if(mDeviceProperties[iTotalDevice].vendor.find("NVIDIA")!=std::string::npos){
					//std::cout<<">> NVIDIA" << std::endl;
					deviceList[iDevice].getInfo(NVIDIA_WAVEFRONT,&(mDeviceProperties[iTotalDevice].warpSize));
				}
				else if(mDeviceProperties[iTotalDevice].vendor.find("AMD")!=std::string::npos){
					//std::cout<<">> AMD" << std::endl;
					deviceList[iDevice].getInfo(AMD_WAVEFRONT,&(mDeviceProperties[iTotalDevice].warpSize));
				}
				else{
					//std::cout<<">> NOT NVIDIA/AMD" << std::endl;
					mDeviceProperties[iTotalDevice].warpSize=16;
				}



				//store the context
				mDeviceProperties[iTotalDevice].oclContext=context;

				//store the device
				mDeviceProperties[iTotalDevice].oclDevice=deviceList[iDevice];

				//store CPU the device
				mDeviceProperties[iTotalDevice].oclDevice=deviceList[0];

				iTotalDevice++;
			}

		}
	}
	catch(const cl::Error &err){
		std::string errString=Utils::OCLErr_code(err.err());
		throw std::runtime_error { errString };
	}
	iCurrentDevice=scelta;

	try{
		mDeviceProperties[iCurrentDevice].oclQueue=cl::CommandQueue(mDeviceProperties[iCurrentDevice].oclContext, mDeviceProperties[iCurrentDevice].oclDevice, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|CL_QUEUE_PROFILING_ENABLE );
/*
		for(int i=0;i<Constants::ITS::TrackletsPerRoad;i++){
			mDeviceProperties[iCurrentDevice].oclCommandQueues[i]=cl::CommandQueue(mDeviceProperties[iCurrentDevice].oclContext, mDeviceProperties[iCurrentDevice].oclDevice, 0);
		}
*/
		//mDeviceProperties[iCurrentDevice].oclCountTrackletKernel=GPU::Utils::CreateKernelFromFile(mDeviceProperties[iCurrentDevice].oclContext,mDeviceProperties[iCurrentDevice].oclDevice,"./src/kernel/countLayerTracklets.cl","countLayerTracklets");
		//mDeviceProperties[iCurrentDevice].oclComputeTrackletKernel=GPU::Utils::CreateKernelFromFile(mDeviceProperties[iCurrentDevice].oclContext,mDeviceProperties[iCurrentDevice].oclDevice,"./src/kernel/computeLayerTracklets.cl","computeLayerTracklets");
		//mDeviceProperties[iCurrentDevice].oclCountCellKernel=GPU::Utils::CreateKernelFromFile(mDeviceProperties[iCurrentDevice].oclContext,mDeviceProperties[iCurrentDevice].oclDevice,"./src/kernel/countLayerCells.cl","countLayerCells");
		//mDeviceProperties[iCurrentDevice].oclComputeCellKernel=GPU::Utils::CreateKernelFromFile(mDeviceProperties[iCurrentDevice].oclContext,mDeviceProperties[iCurrentDevice].oclDevice,"./src/kernel/computeLayerCells.cl","computeLayerCells");
		t2 = clock();
		totalTime = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);
		std::cout<<"ocl-endClock: "<<totalTime<<std::endl;
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
