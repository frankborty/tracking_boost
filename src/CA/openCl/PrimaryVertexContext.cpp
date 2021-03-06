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
#include "ITSReconstruction/CA/gpu/Context.h"
#include "ITSReconstruction/CA/Event.h"


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
}
void PrimaryVertexContext::boostInitialize(
		const Event& event,
		float3 &mPrimaryVertex,
		const std::array<std::vector<Cluster>, Constants::ITS::LayersNumber>& clusters,
		std::array<std::vector<int>, Constants::ITS::CellsPerRoad>& mTrackletsLookupTable
		)
{

	std::array<std::array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>,
			                    Constants::ITS::TrackletsPerRoad> tmpIndexTables;

	compute::context boostContext =GPU::Context::getInstance().getBoostDeviceProperties().boostContext;
	compute::command_queue boostQueue =GPU::Context::getInstance().getBoostDeviceProperties().boostCommandQueues[0];
	u_int iClusterNum,iTrackletNum,iTrackletLookupSize,cellsLookupTableMemorySize;
	int iClusterSize[Constants::ITS::LayersNumber];
	if(iInitialize==1){

		for(int i=0;i<o2::ITS::CA::Constants::ITS::TrackletsPerRoad;i++){
			this-> boostLayerIndex[i]=compute::buffer(boostContext,sizeof(int),(cl_mem_flags)CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,&(i));
		}

		this->boostPrimaryVertex= boost::compute::buffer(boostContext,sizeof(FLOAT3));

		for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer){
			this->boostClusters[iLayer]=compute::vector<Cluster>(1,boostContext);

			if(iLayer < Constants::ITS::TrackletsPerRoad) {
				this->boostTracklets[iLayer]=compute::vector<Tracklet>(1,boostContext);
			}

			if(iLayer < Constants::ITS::CellsPerRoad - 1){
				this->boostCellsLookupTable[iLayer]=compute::vector<int>(1,boostContext);
			}

			if(iLayer > 0){
				this->boostIndexTables[iLayer-1]=compute::vector<int>(Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1,boostContext);
			}
		}



		iInitialize=0;
	}

	boostQueue.enqueue_write_buffer((this->boostPrimaryVertex), 0, sizeof(FLOAT3), &mPrimaryVertex);

	for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer){
		iClusterNum=clusters[iLayer].size();
		iClusterSize[iLayer]=iClusterNum;
		if(this->boostClusters[iLayer].capacity()<=iClusterNum)
			this->boostClusters[iLayer].reserve(iClusterNum);
		compute::copy_n(clusters[iLayer].begin(), iClusterNum, this->boostClusters[iLayer].begin(), boostQueue);

		if(iLayer < Constants::ITS::TrackletsPerRoad) {
			iTrackletNum = std::ceil((Constants::Memory::TrackletsMemoryCoefficients[iLayer] * event.getLayer(iLayer).getClustersSize())
				   * event.getLayer(iLayer + 1).getClustersSize());
			if(this->boostTracklets[iLayer].capacity()<=iTrackletNum)
				this->boostTracklets[iLayer].reserve(iTrackletNum);
		}

		if(iLayer < Constants::ITS::CellsPerRoad) {
			iTrackletLookupSize=event.getLayer(iLayer + 1).getClustersSize();
			this->boostTrackletsLookupTable[iLayer]=compute::vector<int>(iTrackletLookupSize,boostContext);
			compute::fill(this->boostTrackletsLookupTable[iLayer].begin(),this->boostTrackletsLookupTable[iLayer].end(),-1,boostQueue);
		}

		 if(iLayer < Constants::ITS::CellsPerRoad - 1) {
			cellsLookupTableMemorySize=std::ceil((Constants::Memory::TrackletsMemoryCoefficients[iLayer + 1] * event.getLayer(iLayer + 1).getClustersSize())
						* event.getLayer(iLayer + 2).getClustersSize());
			this->boostCellsLookupTable[iLayer].clear();
			if(this->boostCellsLookupTable[iLayer].size()<=cellsLookupTableMemorySize)
				this->boostCellsLookupTable[iLayer].resize(cellsLookupTableMemorySize);
		 }

		 if(iLayer >0){
			int previousBinIndex { 0 };
			tmpIndexTables[iLayer - 1][0] = 0;
			for (int iCluster { 0 }; iCluster < (int)iClusterNum; ++iCluster) {
				const int currentBinIndex { clusters[iLayer][iCluster].indexTableBinIndex };
				if (currentBinIndex > previousBinIndex) {
					for (int iBin { previousBinIndex + 1 }; iBin <= currentBinIndex; ++iBin) {
						tmpIndexTables[iLayer - 1][iBin] = iCluster;
					}
					previousBinIndex = currentBinIndex;
				}
			}

			for (int iBin { previousBinIndex + 1 }; iBin <= Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins;iBin++) {
				tmpIndexTables[iLayer - 1][iBin] = iClusterNum;
			}
			compute::copy(tmpIndexTables[iLayer-1].begin(), tmpIndexTables[iLayer-1].end(), this->boostIndexTables[iLayer-1].begin(), boostQueue);
		 }


	}
	this->boostClusterSize=compute::buffer(boostContext,Constants::ITS::LayersNumber*sizeof(int),(cl_mem_flags)CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,iClusterSize);


}


}
}
}
}

#endif /* _TRAKINGITSU_INCLUDE_PRIMARY_VERTEX_CONTEXT_H_ */
