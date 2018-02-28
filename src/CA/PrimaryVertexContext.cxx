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

#include "ITSReconstruction/CA/PrimaryVertexContext.h"

#include "ITSReconstruction/CA/Event.h"



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
	std::cout<<"initialize primary vertex"<<std::endl;
	mPrimaryVertex = event.getPrimaryVertex(primaryVertexIndex);
	cl::Context oclContext=GPU::Context::getInstance().getDeviceProperties().oclContext;

	mGPUContext.initialize(oclContext);

	float3 mPrimaryVertex=event.getPrimaryVertex(primaryVertexIndex);
	mGPUContext.mPrimaryVertex.x=mPrimaryVertex.x;
	mGPUContext.mPrimaryVertex.y=mPrimaryVertex.y;
	mGPUContext.mPrimaryVertex.z=mPrimaryVertex.z;

	mGPUContext.bPrimaryVertex=cl::Buffer(
			oclContext,
			(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			3*sizeof(float),
			(void *) &(mGPUContext.mPrimaryVertex));

	//clusters
//	t1=clock();
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


		if(mGPUContext.mClusters[iLayer]!=NULL)
			free(mGPUContext.mClusters[iLayer]);

		int clusterSize=clustersNum*sizeof(ClusterStruct);

		mGPUContext.mClusters[iLayer]=(ClusterStruct*)malloc(clustersNum*sizeof(ClusterStruct));
		mGPUContext.iClusterAllocatedSize[iLayer]=clusterSize;
		mGPUContext.iClusterSize[iLayer]=clustersNum;


		for (int iCluster { 0 }; iCluster < clustersNum; ++iCluster) {
			const Cluster& currentCluster { currentLayer.getCluster(iCluster) };
			mGPUContext.addClusters(mPrimaryVertex,currentCluster,iLayer,iCluster);

		}

		mGPUContext.sortClusters(iLayer);

		mGPUContext.bClusters[iLayer]=cl::Buffer(
			oclContext,
			(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			clusterSize,
			(void *) mGPUContext.mClusters[iLayer]);

		if(iLayer < Constants::ITS::CellsPerRoad) {
			if(mGPUContext.mCells[iLayer]!=NULL)
				free(mGPUContext.mCells[iLayer]);


			mCells[iLayer].clear();
			float cellsMemorySize = std::ceil(((Constants::Memory::CellsMemoryCoefficients[iLayer] * event.getLayer(iLayer).getClustersSize())
			 * event.getLayer(iLayer + 1).getClustersSize()) * event.getLayer(iLayer + 2).getClustersSize());
			mCells[iLayer].reserve(cellsMemorySize);

			int cellSize=cellsMemorySize*sizeof(CellStruct);
			mGPUContext.iCellSize[iLayer]=cellsMemorySize;
			mGPUContext.mCells[iLayer]=(CellStruct*)malloc(cellSize);
			mGPUContext.bCells[iLayer]=cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				cellSize,
				(void *) mGPUContext.mCells[iLayer]);
		}

		if(iLayer < Constants::ITS::CellsPerRoad - 1) {
			//mCellsLookupTable[iLayer].clear();
			if(mGPUContext.iCellsLookupTable[iLayer]!=NULL)
				free(mGPUContext.iCellsLookupTable[iLayer]);

			int cellsLookupTableMemorySize=std::ceil((Constants::Memory::TrackletsMemoryCoefficients[iLayer + 1] * event.getLayer(iLayer + 1).getClustersSize())
			* event.getLayer(iLayer + 2).getClustersSize());


			int CellsLookupTableSize=cellsLookupTableMemorySize*sizeof(int);
			mGPUContext.iCellsLookupTableSize[iLayer]=cellsLookupTableMemorySize;
			mGPUContext.iCellsLookupTable[iLayer]=(int*)malloc(CellsLookupTableSize);

			mGPUContext.bCellsLookupTable[iLayer]=cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				CellsLookupTableSize,
				(void *) mGPUContext.iCellsLookupTable[iLayer]);

			if(iLayer < Constants::ITS::CellsPerRoad - 1) {
				mCellsLookupTable[iLayer].clear();
				mCellsLookupTable[iLayer].resize(cellsLookupTableMemorySize);

				mCellsNeighbours[iLayer].clear();
			}
		}


	}
	for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad - 1; ++iLayer) {
		mCellsNeighbours[iLayer].clear();
	  }
	mRoads.clear();

	mGPUContext.bClustersSize=cl::Buffer(
		oclContext,
		(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		7*sizeof(int),
		(void *) mGPUContext.iClusterSize);



	for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {
		const int clustersNum = static_cast<int>(mGPUContext.iClusterSize[iLayer]);

		//index table
		if(iLayer > 0) {
			int previousBinIndex { 0 };
			mGPUContext.mIndexTables[iLayer - 1][0] = 0;

			for (int iCluster { 0 }; iCluster < clustersNum; ++iCluster) {
				const int currentBinIndex { mGPUContext.mClusters[iLayer][iCluster].indexTableBinIndex };
				if (currentBinIndex > previousBinIndex) {
					for (int iBin { previousBinIndex + 1 }; iBin <= currentBinIndex; ++iBin) {
						mGPUContext.mIndexTables[iLayer - 1][iBin] = iCluster;
					}
					previousBinIndex = currentBinIndex;
				}
			}

			for (int iBin { previousBinIndex + 1 }; iBin <= Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins;iBin++) {
				mGPUContext.mIndexTables[iLayer - 1][iBin] = clustersNum;
			}

			mGPUContext.bIndexTables[iLayer-1]=cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				mGPUContext.iIndexTableSize*sizeof(int),
				(void *) mGPUContext.mIndexTables[iLayer-1]);
		}

		//tracklets
		if(iLayer < Constants::ITS::TrackletsPerRoad) {
			if(mGPUContext.mTracklets[iLayer]!=NULL)
				free(mGPUContext.mTracklets[iLayer]);

		  float trackletsMemorySize = std::ceil((Constants::Memory::TrackletsMemoryCoefficients[iLayer] * event.getLayer(iLayer).getClustersSize())
			 * event.getLayer(iLayer + 1).getClustersSize());
		  int trackletSize=trackletsMemorySize*sizeof(TrackletStruct);

		  mGPUContext.mTracklets[iLayer]=(TrackletStruct*)malloc(trackletSize);//delete
		  mGPUContext.bTracklets[iLayer]=cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				trackletSize,
				(void *) mGPUContext.mTracklets[iLayer]);

		}

		//// cells
		if(iLayer < Constants::ITS::CellsPerRoad) {
			if(mGPUContext.mCells[iLayer]!=NULL)
				free(mGPUContext.mCells[iLayer]);


			mCells[iLayer].clear();
			float cellsMemorySize = std::ceil(((Constants::Memory::CellsMemoryCoefficients[iLayer] * event.getLayer(iLayer).getClustersSize())
			 * event.getLayer(iLayer + 1).getClustersSize()) * event.getLayer(iLayer + 2).getClustersSize());
			mCells[iLayer].reserve(cellsMemorySize);

			int cellSize=cellsMemorySize*sizeof(CellStruct);
			mGPUContext.iCellSize[iLayer]=cellsMemorySize;
			mGPUContext.mCells[iLayer]=(CellStruct*)malloc(cellSize);
			mGPUContext.bCells[iLayer]=cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				cellSize,
				(void *) mGPUContext.mCells[iLayer]);

		}


		//tracklets lookup
		if(iLayer < Constants::ITS::CellsPerRoad) {

			if(mGPUContext.mTrackletsLookupTable[iLayer]!=NULL)
				free(mGPUContext.mTrackletsLookupTable[iLayer]);

			int size=event.getLayer(iLayer + 1).getClustersSize();
			int lookUpSize=size*sizeof(int);


			mGPUContext.mTrackletsLookupTable[iLayer]=(int*)malloc(lookUpSize);
			mGPUContext.iTrackletsLookupTableSize[iLayer]=size;
			memset(mGPUContext.mTrackletsLookupTable[iLayer],-1,lookUpSize); //forse devo mettere size

			mGPUContext.bTrackletsLookupTable[iLayer]=cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				lookUpSize,
				(void *) mGPUContext.mTrackletsLookupTable[iLayer]);
		}
	}
//		//// cells
//		if(iLayer < Constants::ITS::CellsPerRoad) {
//			if(openClPrimaryVertexContext.mCells[iLayer]!=NULL)
//				free(openClPrimaryVertexContext.mCells[iLayer]);
//
//
//			mCells[iLayer].clear();
//			float cellsMemorySize = std::ceil(((Constants::Memory::CellsMemoryCoefficients[iLayer] * event.getLayer(iLayer).getClustersSize())
//			 * event.getLayer(iLayer + 1).getClustersSize()) * event.getLayer(iLayer + 2).getClustersSize());
//			mCells[iLayer].reserve(cellsMemorySize);
//
//			int cellSize=cellsMemorySize*sizeof(CellStruct);
//			openClPrimaryVertexContext.iCellSize[iLayer]=cellsMemorySize;
//			openClPrimaryVertexContext.mCells[iLayer]=(CellStruct*)malloc(cellSize);//delete
//			openClPrimaryVertexContext.bCells[iLayer]=cl::Buffer(
//				oclContext,
//				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
//				cellSize,
//				(void *) openClPrimaryVertexContext.mCells[iLayer]);
//
//		}
//
//		if(iLayer < Constants::ITS::CellsPerRoad - 1) {
//			//mCellsLookupTable[iLayer].clear();
//			if(openClPrimaryVertexContext.iCellsLookupTable[iLayer]!=NULL)
//				free(openClPrimaryVertexContext.iCellsLookupTable[iLayer]);
//
//			int cellsLookupTableMemorySize=std::ceil((Constants::Memory::TrackletsMemoryCoefficients[iLayer + 1] * event.getLayer(iLayer + 1).getClustersSize())
//			* event.getLayer(iLayer + 2).getClustersSize());
//
//			if((cellsLookupTableMemorySize % workgroupSize)!=0){
//				int mult=cellsLookupTableMemorySize/workgroupSize;
//				cellsLookupTableMemorySize=(mult+1)*workgroupSize;
//			}
//			int CellsLookupTableSize=cellsLookupTableMemorySize*sizeof(int);
//			openClPrimaryVertexContext.iCellsLookupTableSize[iLayer]=cellsLookupTableMemorySize;
//			openClPrimaryVertexContext.iCellsLookupTable[iLayer]=(int*)malloc(CellsLookupTableSize);
//
//			openClPrimaryVertexContext.bCellsLookupTable[iLayer]=cl::Buffer(
//				oclContext,
//				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
//				CellsLookupTableSize,
//				(void *) openClPrimaryVertexContext.iCellsLookupTable[iLayer]);
//
//			if(iLayer < Constants::ITS::CellsPerRoad - 1) {
//				mCellsLookupTable[iLayer].clear();
//				mCellsLookupTable[iLayer].resize(cellsLookupTableMemorySize);
//
//				mCellsNeighbours[iLayer].clear();
//			}
//		}
//	}
//	mRoads.clear();
//
//	for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {
//		const int clustersNum = static_cast<int>(openClPrimaryVertexContext.iClusterSize[iLayer]);
//
//		//index table
//		if(iLayer > 0) {
//
//			int previousBinIndex { 0 };
//			openClPrimaryVertexContext.mIndexTables[iLayer - 1][0] = 0;
//
//			for (int iCluster { 0 }; iCluster < clustersNum; ++iCluster) {
//				const int currentBinIndex { openClPrimaryVertexContext.mClusters[iLayer][iCluster].indexTableBinIndex };
//				if (currentBinIndex > previousBinIndex) {
//					for (int iBin { previousBinIndex + 1 }; iBin <= currentBinIndex; ++iBin) {
//						openClPrimaryVertexContext.mIndexTables[iLayer - 1][iBin] = iCluster;
//					}
//					previousBinIndex = currentBinIndex;
//				}
//			}
//
//			for (int iBin { previousBinIndex + 1 }; iBin <= Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins;iBin++) {
//				openClPrimaryVertexContext.mIndexTables[iLayer - 1][iBin] = clustersNum;
//			}
//
//			openClPrimaryVertexContext.bIndexTables[iLayer-1]=cl::Buffer(
//				oclContext,
//				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
//				openClPrimaryVertexContext.iIndexTableSize*sizeof(int),
//				(void *) openClPrimaryVertexContext.mIndexTables[iLayer-1]);
//
//
//		}
//		//tracklets
//		if(iLayer < Constants::ITS::TrackletsPerRoad) {
//			if(openClPrimaryVertexContext.mTracklets[iLayer]!=NULL)
//				free(openClPrimaryVertexContext.mTracklets[iLayer]);
//
//		  float trackletsMemorySize = std::ceil((Constants::Memory::TrackletsMemoryCoefficients[iLayer] * event.getLayer(iLayer).getClustersSize())
//			 * event.getLayer(iLayer + 1).getClustersSize());
//		  int trackletSize=trackletsMemorySize*sizeof(TrackletStruct);
//		/*  int factor=trackletSize%64;
//		  if(factor!=0){
//			factor++;
//			trackletSize=factor*trackletSize;
//		  }
///*
//		  int res=posix_memalign((void**)&openClPrimaryVertexContext.mTracklets[iLayer],4096,trackletSize);
//		  if(res!=0){
//			  std::cout<<"layer = "<<iLayer<<"\t trackletSize result = "<<res<<std::endl;
//		  }
//*/
//		  openClPrimaryVertexContext.mTracklets[iLayer]=(TrackletStruct*)malloc(trackletSize);//delete
//		  openClPrimaryVertexContext.bTracklets[iLayer]=cl::Buffer(
//				oclContext,
//				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
//				trackletSize,
//				(void *) openClPrimaryVertexContext.mTracklets[iLayer]);
//
//		}
//
//		//tracklets lookup
//		if(iLayer < Constants::ITS::CellsPerRoad) {
//			int workgroupSize=5*32;
//			if(openClPrimaryVertexContext.mTrackletsLookupTable[iLayer]!=NULL)
//				free(openClPrimaryVertexContext.mTrackletsLookupTable[iLayer]);
//			int size=event.getLayer(iLayer + 1).getClustersSize()*sizeof(int);
//			/*if((size % workgroupSize)!=0){
//				int mult=size/workgroupSize;
//				size=(mult+1)*workgroupSize;
//			}*/
//			int lookUpSize=size;
//			/*int factor=lookUpSize%64;
//			if(factor!=0){
//				factor++;
//				lookUpSize=factor*lookUpSize;
//			}*/
//			if((lookUpSize % workgroupSize)!=0){
//				int mult=lookUpSize/workgroupSize;
//				lookUpSize=(mult+1)*workgroupSize;
//			}
//			//openClPrimaryVertexContext.mTrackletsLookupTable[iLayer]=(int*)(4096,lookUpSize);
//			openClPrimaryVertexContext.mTrackletsLookupTable[iLayer]=(int*)malloc(lookUpSize);
//			//int res=posix_memalign((void**)&openClPrimaryVertexContext.mTrackletsLookupTable[iLayer],4096,lookUpSize);
//			//if(res!=0){
//			//	std::cout<<"layer = "<<iLayer<<"\t tracklets lookup = "<<res<<std::endl;
//			//}
//			openClPrimaryVertexContext.iTrackletsLookupTableAllocatedSize[iLayer]=lookUpSize;
//			openClPrimaryVertexContext.iTrackletsLookupTableSize[iLayer]=size;
//			memset(openClPrimaryVertexContext.mTrackletsLookupTable[iLayer],-1,lookUpSize);
//
//			openClPrimaryVertexContext.bTrackletsLookupTable[iLayer]=cl::Buffer(
//				oclContext,
//				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
//				lookUpSize,
//				(void *) openClPrimaryVertexContext.mTrackletsLookupTable[iLayer]);
//		}
//	}
//	openClPrimaryVertexContext.bClustersSize=cl::Buffer(
//					oclContext,
//					(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
//					7*sizeof(int),
//					(void *) openClPrimaryVertexContext.iClusterSize);
//*/



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
