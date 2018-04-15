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

#include <ITSReconstruction/CA/Cluster.h>
#include <ITSReconstruction/CA/Layer.h>
#include <ITSReconstruction/CA/PrimaryVertexContext.h>
#include <cmath>



namespace o2
{
namespace ITS
{
namespace CA
{

PrimaryVertexContext::PrimaryVertexContext()
{
  // Nothing to do
}

void PrimaryVertexContext::initialize(const Event& event, const int primaryVertexIndex) {

#if TRACKINGITSU_OCL_MODE
	//std::cout<<"initialize OCL primary vertex"<<std::endl;
	mPrimaryVertex = event.getPrimaryVertex(primaryVertexIndex);
	compute::context boostContext =GPU::Context::getInstance().getBoostDeviceProperties().boostContext;
	compute::command_queue boostQueue =GPU::Context::getInstance().getBoostDeviceProperties().boostCommandQueues[0];
	mGPUContext.boostInitialize(boostContext,boostQueue);

	mGPUContext.boostPrimaryVertex=compute::buffer(boostContext,sizeof(float3),(cl_mem_flags)CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,&(mPrimaryVertex));

	for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {

	    const Layer& currentLayer { event.getLayer(iLayer) };
	    const int clustersNum { currentLayer.getClustersSize() };
	    mGPUContext.iClusterSize[iLayer]=clustersNum;
	    mClusters[iLayer].clear();
	    if(mGPUContext.boostClusters[iLayer].empty()==false)
	    	mGPUContext.boostClusters[iLayer].clear();

	    if(clustersNum > static_cast<int>(mClusters[iLayer].capacity())) {
	      mClusters[iLayer].reserve(clustersNum);
	    }

	    for (int iCluster { 0 }; iCluster < clustersNum; ++iCluster) {

	      const Cluster& currentCluster { currentLayer.getCluster(iCluster) };
	      mClusters[iLayer].emplace_back(iLayer, event.getPrimaryVertex(primaryVertexIndex), currentCluster);
	    }

	    std::sort(mClusters[iLayer].begin(), mClusters[iLayer].end(), [](Cluster& cluster1, Cluster& cluster2) {
	      return cluster1.indexTableBinIndex < cluster2.indexTableBinIndex;
	    });


		mGPUContext.boostClusters[iLayer]=compute::vector<Cluster>(clustersNum*sizeof(Cluster),boostContext);
		compute::copy(mClusters[iLayer].begin(), mClusters[iLayer].end(), mGPUContext.boostClusters[iLayer].begin(), boostQueue);

	    if(iLayer < Constants::ITS::CellsPerRoad) {
	      mCells[iLayer].clear();
	      float cellsMemorySize = std::ceil(((Constants::Memory::CellsMemoryCoefficients[iLayer] * event.getLayer(iLayer).getClustersSize())
	         * event.getLayer(iLayer + 1).getClustersSize()) * event.getLayer(iLayer + 2).getClustersSize());

	      if(cellsMemorySize > mCells[iLayer].capacity()) {
	        mCells[iLayer].reserve(cellsMemorySize);
	      }
	      mGPUContext.boostCells[iLayer]=compute::vector<Cell>(cellsMemorySize,boostContext);
	      //compute::copy(mCells[iLayer].begin(), mCells[iLayer].end(), mGPUContext.boostCells[iLayer].begin(), boostQueue);
	    }

	    if(iLayer < Constants::ITS::CellsPerRoad - 1) {
	    	int cellsLookupTableMemorySize=std::ceil((Constants::Memory::TrackletsMemoryCoefficients[iLayer + 1] * event.getLayer(iLayer + 1).getClustersSize())
	    				* event.getLayer(iLayer + 2).getClustersSize());
	      mCellsLookupTable[iLayer].clear();
	      mCellsLookupTable[iLayer].resize(cellsLookupTableMemorySize, Constants::ITS::UnusedIndex);


	      mGPUContext.boostCellsLookupTable[iLayer]=compute::vector<int>(cellsLookupTableMemorySize,boostContext);
	      compute::copy(mCellsLookupTable[iLayer].begin(), mCellsLookupTable[iLayer].end(), mGPUContext.boostCellsLookupTable[iLayer].begin(), boostQueue);


	      mCellsNeighbours[iLayer].clear();
	    }
	  }

	  for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad - 1; ++iLayer) {
	    mCellsNeighbours[iLayer].clear();
	  }

	  mRoads.clear();
	  for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {
	      const int clustersNum = static_cast<int>(mClusters[iLayer].size());
	      if(iLayer > 0) {
	        int previousBinIndex { 0 };
	        mIndexTables[iLayer - 1][0] = 0;
	        for (int iCluster { 0 }; iCluster < clustersNum; ++iCluster) {
	          const int currentBinIndex { mClusters[iLayer][iCluster].indexTableBinIndex };
	          if (currentBinIndex > previousBinIndex) {
	            for (int iBin { previousBinIndex + 1 }; iBin <= currentBinIndex; ++iBin) {
	              mIndexTables[iLayer - 1][iBin] = iCluster;
	            }
	            previousBinIndex = currentBinIndex;
	          }
	        }

	        for (int iBin { previousBinIndex + 1 }; iBin <= Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins;
	            iBin++) {
	          mIndexTables[iLayer - 1][iBin] = clustersNum;
	        }

	        mGPUContext.boostIndexTables[iLayer-1]=compute::buffer(
					boostContext,
					mGPUContext.iIndexTableSize*sizeof(int),
					(cl_mem_flags)CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					(void *) &(mIndexTables[iLayer-1][0]));
	      }
	      if(iLayer < Constants::ITS::TrackletsPerRoad) {
	        mTracklets[iLayer].clear();
	        float trackletsMemorySize = std::ceil((Constants::Memory::TrackletsMemoryCoefficients[iLayer] * event.getLayer(iLayer).getClustersSize())
	           * event.getLayer(iLayer + 1).getClustersSize());

	        if(trackletsMemorySize > mTracklets[iLayer].capacity()) {
	          mTracklets[iLayer].reserve(trackletsMemorySize);
	        }
	        mGPUContext.boostTracklets[iLayer]=compute::vector<Tracklet>(trackletsMemorySize,boostContext);
			//compute::copy(mTracklets[iLayer].begin(), mTracklets[iLayer].end(), mGPUContext.boostTracklets[iLayer].begin(), boostQueue);
	      }

	      if(iLayer < Constants::ITS::CellsPerRoad) {
	        mTrackletsLookupTable[iLayer].clear();
	        mTrackletsLookupTable[iLayer].resize(
	           event.getLayer(iLayer + 1).getClustersSize(), Constants::ITS::UnusedIndex);
	        int size=event.getLayer(iLayer + 1).getClustersSize();

	        mGPUContext.boostTrackletsLookupTable[iLayer]=compute::vector<int>(size,boostContext);
	        compute::copy(mTrackletsLookupTable[iLayer].begin(), mTrackletsLookupTable[iLayer].end(), mGPUContext.boostTrackletsLookupTable[iLayer].begin(), boostQueue);
	      }
	    }

	  	mGPUContext.boostClusterSize=compute::buffer(boostContext,Constants::ITS::LayersNumber*sizeof(int),(cl_mem_flags)CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,mGPUContext.iClusterSize);


#endif

#if	!TRACKINGITSU_GPU_MODE
	mPrimaryVertex = event.getPrimaryVertex(primaryVertexIndex);

  for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {

    const Layer& currentLayer { event.getLayer(iLayer) };
    const int clustersNum { currentLayer.getClustersSize() };

    mClusters[iLayer].clear();

    if(clustersNum > static_cast<int>(mClusters[iLayer].capacity())) {

      mClusters[iLayer].reserve(clustersNum);
    }

    for (int iCluster { 0 }; iCluster < clustersNum; ++iCluster) {

      const Cluster& currentCluster { currentLayer.getCluster(iCluster) };
      mClusters[iLayer].emplace_back(iLayer, event.getPrimaryVertex(primaryVertexIndex), currentCluster);
    }

    std::sort(mClusters[iLayer].begin(), mClusters[iLayer].end(), [](Cluster& cluster1, Cluster& cluster2) {
      return cluster1.indexTableBinIndex < cluster2.indexTableBinIndex;
    });

    if(iLayer < Constants::ITS::CellsPerRoad) {

      mCells[iLayer].clear();
      float cellsMemorySize = std::ceil(((Constants::Memory::CellsMemoryCoefficients[iLayer] * event.getLayer(iLayer).getClustersSize())
         * event.getLayer(iLayer + 1).getClustersSize()) * event.getLayer(iLayer + 2).getClustersSize());

      if(cellsMemorySize > mCells[iLayer].capacity()) {

        mCells[iLayer].reserve(cellsMemorySize);
      }
    }

    if(iLayer < Constants::ITS::CellsPerRoad - 1) {

      mCellsLookupTable[iLayer].clear();
      mCellsLookupTable[iLayer].resize(std::ceil(
        (Constants::Memory::TrackletsMemoryCoefficients[iLayer + 1] * event.getLayer(iLayer + 1).getClustersSize())
          * event.getLayer(iLayer + 2).getClustersSize()), Constants::ITS::UnusedIndex);


      mCellsNeighbours[iLayer].clear();
    }
  }

  for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad - 1; ++iLayer) {

    mCellsNeighbours[iLayer].clear();
  }

  mRoads.clear();
#endif

 #if TRACKINGITSU_CUDA_MODE
  	  mGPUContextDevicePointer = mGPUContext.initialize(mPrimaryVertex, mClusters, mCells, mCellsLookupTable);
#endif

#if !TRACKINGITSU_GPU_MODE
  for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {

    const int clustersNum = static_cast<int>(mClusters[iLayer].size());

    if(iLayer > 0) {

      int previousBinIndex { 0 };
      mIndexTables[iLayer - 1][0] = 0;

      for (int iCluster { 0 }; iCluster < clustersNum; ++iCluster) {

        const int currentBinIndex { mClusters[iLayer][iCluster].indexTableBinIndex };

        if (currentBinIndex > previousBinIndex) {

          for (int iBin { previousBinIndex + 1 }; iBin <= currentBinIndex; ++iBin) {

            mIndexTables[iLayer - 1][iBin] = iCluster;
          }

          previousBinIndex = currentBinIndex;
        }
      }

      for (int iBin { previousBinIndex + 1 }; iBin <= Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins;
          iBin++) {

        mIndexTables[iLayer - 1][iBin] = clustersNum;
      }
    }

    if(iLayer < Constants::ITS::TrackletsPerRoad) {

      mTracklets[iLayer].clear();

      float trackletsMemorySize = std::ceil((Constants::Memory::TrackletsMemoryCoefficients[iLayer] * event.getLayer(iLayer).getClustersSize())
         * event.getLayer(iLayer + 1).getClustersSize());

      if(trackletsMemorySize > mTracklets[iLayer].capacity()) {

        mTracklets[iLayer].reserve(trackletsMemorySize);
      }
    }

    if(iLayer < Constants::ITS::CellsPerRoad) {

      mTrackletsLookupTable[iLayer].clear();
      mTrackletsLookupTable[iLayer].resize(
         event.getLayer(iLayer + 1).getClustersSize(), Constants::ITS::UnusedIndex);
    }
  }
#endif
}

}
}
}
