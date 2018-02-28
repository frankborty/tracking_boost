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
/// \file PrimaryVertexContext.cxx
/// \brief
///

#ifndef _TRAKINGITSU_INCLUDE_PRIMARY_VERTEX_CONTEXT_H_
#define _TRAKINGITSU_INCLUDE_PRIMARY_VERTEX_CONTEXT_H_


#include <sstream>
#include <iostream>
#include <algorithm>
#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/gpu/PrimaryVertexContext.h"

namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{

bool myClusterComparator(const ClusterStruct &lhs, const ClusterStruct &rhs)
{
        return lhs.indexTableBinIndex < rhs.indexTableBinIndex;
}


PrimaryVertexContext::PrimaryVertexContext()
{
  // Nothing to do
}

void  PrimaryVertexContext::sortClusters(int iLayer)
{
	std::sort(this->mClusters[iLayer],this->mClusters[iLayer]+this->iClusterSize[iLayer],myClusterComparator);
}


void PrimaryVertexContext::initialize(cl::Context oclContext)
{
	for(int i=0;i<Constants::ITS::LayersNumber;i++){
		this->bLayerIndex[i]=cl::Buffer(oclContext,
		(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(int),
		(void *) &(i));
	}
	int iTrackletsFoundForLayer[]={0,0,0,0,0,0};
	this->bTrackletsFoundForLayer=cl::Buffer(
		oclContext,
		(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		o2::ITS::CA::Constants::ITS::TrackletsPerRoad*sizeof(int),
		(void *) iTrackletsFoundForLayer);

	int iCellsFoundForLayer[]={0,0,0,0,0};
	this->bCellsFoundForLayer=cl::Buffer(
		oclContext,
		(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		o2::ITS::CA::Constants::ITS::CellsPerRoad*sizeof(int),
		(void *) iCellsFoundForLayer);
}

}
}
}
}

#endif /* _TRAKINGITSU_INCLUDE_PRIMARY_VERTEX_CONTEXT_H_ */
