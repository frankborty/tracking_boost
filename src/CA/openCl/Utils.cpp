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
/// \file CAGPUtils.cu
/// \brief
///
#include <string>
#include <vector>
#include "ITSReconstruction/CA/openCl/Utils.h"
#include "ITSReconstruction/CA/gpu/Context.h"

#include <sstream>
#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h>


namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{

int Utils::findNearestDivisor(const int numToRound, const int divisor)
{
	if (numToRound > divisor) {
		return divisor;
	}
	int result = numToRound;

	while (divisor % result != 0) {
		++result;
	}
	return result;
}

int Utils::roundUp(const int numToRound, const int multiple)
{
	if (multiple == 0) {
		return numToRound;
	}
	int remainder { numToRound % multiple };
	if (remainder == 0) {
		return numToRound;
	}
	return numToRound + multiple - remainder;
}

cl::Kernel Utils::CreateKernelFromFile(cl::Context oclContext, cl::Device oclDevice, const char* fileName,const char* kernelName){
	//std::cout << "CreateKernelFromFile: "<<fileName <<"... ";
	cl::Kernel kernel;
	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "\nFailed to open file for reading: " << fileName << std::endl;
		return cl::Kernel();
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();
	//std::cerr<<srcStr<< std::endl;

	cl::Program::Sources sources;
	sources.push_back({srcStdStr.c_str(),srcStdStr.length()});

	cl::Program program(oclContext,sources);
	try{
		std::vector<cl::Device> oclDeviceList;
		oclDeviceList.push_back(oclDevice);
		char buildOption[100];
		sprintf(buildOption,"-cl-std=CL2.0");
		//program.build({oclDevice},buildOption);
		program.build({oclDevice});
		kernel=cl::Kernel(program,kernelName);
	}
	catch(const cl::Error &err){

		std::string errString=Utils::OCLErr_code(err.err());
		std::cout<< errString << std::endl;

		std::cerr
				<< "OpenCL compilation error" << std::endl
				<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(oclDevice)
				<< std::endl;

		throw std::runtime_error { errString };
	}
	return kernel;
}




char* Utils::OCLErr_code (int err_in){
	switch (err_in) {

	case CL_SUCCESS :
		return (char*)" CL_SUCCESS ";
	case CL_DEVICE_NOT_FOUND :
		return (char*)" CL_DEVICE_NOT_FOUND ";
	case CL_DEVICE_NOT_AVAILABLE :
		return (char*)" CL_DEVICE_NOT_AVAILABLE ";
	case CL_COMPILER_NOT_AVAILABLE :
		return (char*)" CL_COMPILER_NOT_AVAILABLE ";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE :
		return (char*)" CL_MEM_OBJECT_ALLOCATION_FAILURE ";
	case CL_OUT_OF_RESOURCES :
		return (char*)" CL_OUT_OF_RESOURCES ";
	case CL_OUT_OF_HOST_MEMORY :
		return (char*)" CL_OUT_OF_HOST_MEMORY ";
	case CL_PROFILING_INFO_NOT_AVAILABLE :
		return (char*)" CL_PROFILING_INFO_NOT_AVAILABLE ";
	case CL_MEM_COPY_OVERLAP :
		return (char*)" CL_MEM_COPY_OVERLAP ";
	case CL_IMAGE_FORMAT_MISMATCH :
		return (char*)" CL_IMAGE_FORMAT_MISMATCH ";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED :
		return (char*)" CL_IMAGE_FORMAT_NOT_SUPPORTED ";
	case CL_BUILD_PROGRAM_FAILURE :
		return (char*)" CL_BUILD_PROGRAM_FAILURE ";
	case CL_MAP_FAILURE :
		return (char*)" CL_MAP_FAILURE ";
	case CL_MISALIGNED_SUB_BUFFER_OFFSET :
		return (char*)" CL_MISALIGNED_SUB_BUFFER_OFFSET ";
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST :
		return (char*)" CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ";
	case CL_INVALID_VALUE :
		return (char*)" CL_INVALID_VALUE ";
	case CL_INVALID_DEVICE_TYPE :
		return (char*)" CL_INVALID_DEVICE_TYPE ";
	case CL_INVALID_PLATFORM :
		return (char*)" CL_INVALID_PLATFORM ";
	case CL_INVALID_DEVICE :
		return (char*)" CL_INVALID_DEVICE ";
	case CL_INVALID_CONTEXT :
		return (char*)" CL_INVALID_CONTEXT ";
	case CL_INVALID_QUEUE_PROPERTIES :
		return (char*)" CL_INVALID_QUEUE_PROPERTIES ";
	case CL_INVALID_COMMAND_QUEUE :
		return (char*)" CL_INVALID_COMMAND_QUEUE ";
	case CL_INVALID_HOST_PTR :
		return (char*)" CL_INVALID_HOST_PTR ";
	case CL_INVALID_MEM_OBJECT :
		return (char*)" CL_INVALID_MEM_OBJECT ";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR :
		return (char*)" CL_INVALID_IMAGE_FORMAT_DESCRIPTOR ";
	case CL_INVALID_IMAGE_SIZE :
		return (char*)" CL_INVALID_IMAGE_SIZE ";
	case CL_INVALID_SAMPLER :
		return (char*)" CL_INVALID_SAMPLER ";
	case CL_INVALID_BINARY :
		return (char*)" CL_INVALID_BINARY ";
	case CL_INVALID_BUILD_OPTIONS :
		return (char*)" CL_INVALID_BUILD_OPTIONS ";
	case CL_INVALID_PROGRAM :
		return (char*)" CL_INVALID_PROGRAM ";
	case CL_INVALID_PROGRAM_EXECUTABLE :
		return (char*)" CL_INVALID_PROGRAM_EXECUTABLE ";
	case CL_INVALID_KERNEL_NAME :
		return (char*)" CL_INVALID_KERNEL_NAME ";
	case CL_INVALID_KERNEL_DEFINITION :
		return (char*)" CL_INVALID_KERNEL_DEFINITION ";
	case CL_INVALID_KERNEL :
		return (char*)" CL_INVALID_KERNEL ";
	case CL_INVALID_ARG_INDEX :
		return (char*)" CL_INVALID_ARG_INDEX ";
	case CL_INVALID_ARG_VALUE :
		return (char*)" CL_INVALID_ARG_VALUE ";
	case CL_INVALID_ARG_SIZE :
		return (char*)" CL_INVALID_ARG_SIZE ";
	case CL_INVALID_KERNEL_ARGS :
		return (char*)" CL_INVALID_KERNEL_ARGS ";
	case CL_INVALID_WORK_DIMENSION :
		return (char*)" CL_INVALID_WORK_DIMENSION ";
	case CL_INVALID_WORK_GROUP_SIZE :
		return (char*)" CL_INVALID_WORK_GROUP_SIZE ";
	case CL_INVALID_WORK_ITEM_SIZE :
		return (char*)" CL_INVALID_WORK_ITEM_SIZE ";
	case CL_INVALID_GLOBAL_OFFSET :
		return (char*)" CL_INVALID_GLOBAL_OFFSET ";
	case CL_INVALID_EVENT_WAIT_LIST :
		return (char*)" CL_INVALID_EVENT_WAIT_LIST ";
	case CL_INVALID_EVENT :
		return (char*)" CL_INVALID_EVENT ";
	case CL_INVALID_OPERATION :
		return (char*)" CL_INVALID_OPERATION ";
	case CL_INVALID_GL_OBJECT :
		return (char*)" CL_INVALID_GL_OBJECT ";
	case CL_INVALID_BUFFER_SIZE :
		return (char*)" CL_INVALID_BUFFER_SIZE ";
	case CL_INVALID_MIP_LEVEL :
		return (char*)" CL_INVALID_MIP_LEVEL ";
	case CL_INVALID_GLOBAL_WORK_SIZE :
		return (char*)" CL_INVALID_GLOBAL_WORK_SIZE ";
	case CL_INVALID_PROPERTY :
		return (char*)" CL_INVALID_PROPERTY ";
	default:
		return (char*)"UNKNOWN ERROR";

	}
}


}
}
}
}
