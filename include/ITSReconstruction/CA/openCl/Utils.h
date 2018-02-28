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

#ifndef TRACKINGITSU_INCLUDE_OCL_UTILS_H_
#define TRACKINGITSU_INCLUDE_OCL_UTILS_H_

#include "ITSReconstruction/CA/Definitions.h"

namespace o2
{
	namespace ITS
	{
		namespace CA
		{
			namespace GPU
			{
				namespace Utils {
					int findNearestDivisor(const int numToRound, const int divisor);
					int roundUp(const int numToRound, const int multiple);
					char *OCLErr_code (int err_in);
					cl::Kernel CreateKernelFromFile(cl::Context, cl::Device device, const char* fileName, const char* kernelName);
				}
			}
		}
	}
}

#endif /* TRACKINGITSU_INCLUDE_OCL_UTILS_H_ */
