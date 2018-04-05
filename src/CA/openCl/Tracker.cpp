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
///


#include <unistd.h>

#include "ITSReconstruction/CA/Definitions.h"
#include <ITSReconstruction/CA/Tracklet.h>
#include <ITSReconstruction/CA/Cell.h>
#include <ITSReconstruction/CA/Constants.h>
#include <ITSReconstruction/CA/Tracker.h>
#include <ITSReconstruction/CA/Tracklet.h>
#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/gpu/Vector.h"
#include "ITSReconstruction/CA/gpu/Utils.h"


namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{



void computeLayerTracklets(PrimaryVertexContext &primaryVertexContext, const int layerIndex,
    Vector<Tracklet>& trackletsVector)
{

}

void computeLayerCells(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
    Vector<Cell>& cellsVector)
{
}

void layerTrackletsKernel(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
    Vector<Tracklet> trackletsVector)
{
  computeLayerTracklets(primaryVertexContext, layerIndex, trackletsVector);
}

void sortTrackletsKernel(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
    Vector<Tracklet> tempTrackletArray)
{

}

void layerCellsKernel(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
    Vector<Cell> cellsVector)
{
  computeLayerCells(primaryVertexContext, layerIndex, cellsVector);
}



void sortCellsKernel(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
    Vector<Cell> tempCellsArray)
{

}

} /// End of GPU namespace


template<>
void TrackerTraits<true>::computeLayerTracklets(CA::PrimaryVertexContext& primaryVertexContext)
{
	int iClustersNum;

	compute::buffer boostTrackletLookUpTable;

	int workgroupSize=5*32;	//tmp value
	try{
		cl::Context oclContext=GPU::Context::getInstance().getDeviceProperties().oclContext;
		cl::Device oclDevice=GPU::Context::getInstance().getDeviceProperties().oclDevice;
		cl::CommandQueue oclCommandQueue=GPU::Context::getInstance().getDeviceProperties().oclQueue;

	//boost_compute
		compute::device boostDevice = GPU::Context::getInstance().getBoostDeviceProperties().boostDevice;
		compute::context boostContext= GPU::Context::getInstance().getBoostDeviceProperties().boostContext;
		compute::command_queue boostQueue =GPU::Context::getInstance().getBoostDeviceProperties().boostCommandQueue;
		compute::kernel countTrackletKernel=GPU::Context::getInstance().getBoostDeviceProperties().countTrackletsBoostKernel;
		compute::kernel computeTrackletKernel=GPU::Context::getInstance().getBoostDeviceProperties().computeTrackletsBoostKernel;
	///


		iClustersNum=primaryVertexContext.mClusters[0].size();

		if((iClustersNum % workgroupSize)!=0){
			int mult=iClustersNum/workgroupSize;
			iClustersNum=(mult+1)*workgroupSize;
		}

		int *firstLayerLookUpTable;
		firstLayerLookUpTable=(int*)malloc(iClustersNum*sizeof(int));
		memset(firstLayerLookUpTable,-1,iClustersNum*sizeof(int));

		boost::compute::vector<int> boostFirstLayerTrackletsLookup(iClustersNum,boostContext);

		// copy from the host to the device
		boost::compute::copy(
				firstLayerLookUpTable, firstLayerLookUpTable+iClustersNum, boostFirstLayerTrackletsLookup.begin(), boostQueue
		);


		for (int iLayer{ 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer){
			iClustersNum=primaryVertexContext.mClusters[iLayer].size();
			countTrackletKernel.set_arg(0,primaryVertexContext.mGPUContext.boostPrimaryVertex);
			countTrackletKernel.set_arg(1,primaryVertexContext.mGPUContext.boostClusters[iLayer]);
			countTrackletKernel.set_arg(2,primaryVertexContext.mGPUContext.boostClusters[iLayer+1]);
			countTrackletKernel.set_arg(3,primaryVertexContext.mGPUContext.boostIndexTables[iLayer]);
			countTrackletKernel.set_arg(4,primaryVertexContext.mGPUContext.boostLayerIndex[iLayer]);
			countTrackletKernel.set_arg(5,primaryVertexContext.mGPUContext.boostClusterSize);
			if(iLayer==0)
				countTrackletKernel.set_arg(6, boostFirstLayerTrackletsLookup);
			else
				countTrackletKernel.set_arg(6, primaryVertexContext.mGPUContext.boostTrackletsLookupTable[iLayer-1]);


			int pseudoClusterNumber=iClustersNum;
			if((iClustersNum % workgroupSize)!=0){
				int mult=iClustersNum/workgroupSize;
				pseudoClusterNumber=(mult+1)*workgroupSize;
			}
			try{
				boostQueue.enqueue_1d_range_kernel(countTrackletKernel,0,pseudoClusterNumber,0);

			}catch(boost::compute::opencl_error &e){
					    std::cout << e.error_string() << std::endl;
			}catch (std::exception& e) {
				std::cout<<e.what()<<std::endl;
			}
		}

		boostQueue.finish();
		//scan
		for (int iLayer { 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer) {
			iClustersNum=primaryVertexContext.mGPUContext.iClusterSize[iLayer];

			if(iLayer==0){

				// sort data on the device
				compute::inclusive_scan(
						boostFirstLayerTrackletsLookup.begin(),
						boostFirstLayerTrackletsLookup.end(),
						boostFirstLayerTrackletsLookup.begin(),
						boostQueue);

				// copy data back to the host
				compute::copy(
						boostFirstLayerTrackletsLookup.begin(),
						boostFirstLayerTrackletsLookup.end(),
						firstLayerLookUpTable,
						boostQueue);

				primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer]=firstLayerLookUpTable[iClustersNum-1];

			}
			else{
				// sort data on the device
				compute::inclusive_scan(
						primaryVertexContext.mGPUContext.boostTrackletsLookupTable[iLayer-1].begin(),
						primaryVertexContext.mGPUContext.boostTrackletsLookupTable[iLayer-1].end(),
						primaryVertexContext.mGPUContext.boostTrackletsLookupTable[iLayer-1].begin(),
						boostQueue);
				// copy data back to the host
				compute::copy(
						primaryVertexContext.mGPUContext.boostTrackletsLookupTable[iLayer-1].begin(),
						primaryVertexContext.mGPUContext.boostTrackletsLookupTable[iLayer-1].end(),
						&(primaryVertexContext.mTrackletsLookupTable[iLayer-1][0]),
						boostQueue);

				primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer]=primaryVertexContext.mTrackletsLookupTable[iLayer-1][iClustersNum-1];

			}

		}

		primaryVertexContext.mGPUContext.boostTrackletsFoundForLayer=compute::buffer(
			boostContext,
			o2::ITS::CA::Constants::ITS::TrackletsPerRoad*sizeof(int),
			(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			(void *) &(primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[0]));

		//calcolo le tracklet
		for (int iLayer{ 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer) {
			iClustersNum=primaryVertexContext.mGPUContext.iClusterSize[iLayer];
			computeTrackletKernel.set_arg(0,primaryVertexContext.mGPUContext.boostPrimaryVertex);
			computeTrackletKernel.set_arg(5,primaryVertexContext.mGPUContext.boostLayerIndex[iLayer]);

			computeTrackletKernel.set_arg(1,primaryVertexContext.mGPUContext.boostClusters[iLayer]);
			computeTrackletKernel.set_arg(2,primaryVertexContext.mGPUContext.boostClusters[iLayer+1]);
			computeTrackletKernel.set_arg(3,primaryVertexContext.mGPUContext.boostIndexTables[iLayer]);
			computeTrackletKernel.set_arg(4,primaryVertexContext.mGPUContext.boostTracklets[iLayer]);

			computeTrackletKernel.set_arg(6,primaryVertexContext.mGPUContext.boostClusterSize);
			if(iLayer==0)
				computeTrackletKernel.set_arg(7, boostFirstLayerTrackletsLookup);
			else
				computeTrackletKernel.set_arg(7, primaryVertexContext.mGPUContext.boostTrackletsLookupTable[iLayer-1]);

			int pseudoClusterNumber=iClustersNum;
			if((iClustersNum % workgroupSize)!=0){
				int mult=iClustersNum/workgroupSize;
				pseudoClusterNumber=(mult+1)*workgroupSize;
			}

			try{
				boostQueue.enqueue_1d_range_kernel(computeTrackletKernel,0,pseudoClusterNumber,0);

			}catch(boost::compute::opencl_error &e){
						std::cout << e.error_string() << std::endl;
			}catch (std::exception& e) {
				std::cout<<e.what()<<std::endl;
			}

		}

		boostQueue.finish();
		free(firstLayerLookUpTable);
	}catch (std::exception& e) {
			std::cout<<e.what()<<std::endl;
			throw std::runtime_error { e.what() };

	}catch (...) {
		std::cout<<"Exception during compute tracklet phase"<<std::endl;
		throw std::runtime_error { "Exception during compute tracklet phase" };
	}
}

template<>
void TrackerTraits<true>::computeLayerCells(CA::PrimaryVertexContext& primaryVertexContext)
{
	int iTrackletsNum;
	compute::buffer boostTrackletLookUpTable;
	int workgroupSize=5*32;
	int cellsFound[Constants::ITS::CellsPerRoad];
	try{
		//boost_compute
		compute::device boostDevice = GPU::Context::getInstance().getBoostDeviceProperties().boostDevice;
		compute::context boostContext= GPU::Context::getInstance().getBoostDeviceProperties().boostContext;
		compute::command_queue boostQueue =GPU::Context::getInstance().getBoostDeviceProperties().boostCommandQueue;
		compute::kernel countCellsKernel=GPU::Context::getInstance().getBoostDeviceProperties().countCellsBoostKernel;
		compute::kernel computeCellsKernel=GPU::Context::getInstance().getBoostDeviceProperties().computeCellsBoostKernel;
		///


		iTrackletsNum=primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[0];
		int pseudoTrackletsNumber=iTrackletsNum;
		if((pseudoTrackletsNumber % workgroupSize)!=0){
			int mult=pseudoTrackletsNumber/workgroupSize;
			pseudoTrackletsNumber=(mult+1)*workgroupSize;
		}

		int *firstLayerLookUpTable;
		firstLayerLookUpTable=(int*)malloc(pseudoTrackletsNumber*sizeof(int));
		memset(firstLayerLookUpTable,-1,pseudoTrackletsNumber*sizeof(int));

		boost::compute::vector<int> boostFirstLayerCellsLookup(pseudoTrackletsNumber,boostContext);

		// copy from the host to the device
		boost::compute::copy(
				firstLayerLookUpTable, firstLayerLookUpTable+pseudoTrackletsNumber, boostFirstLayerCellsLookup.begin(), boostQueue
		);



		for (int iLayer{ 0 }; iLayer<Constants::ITS::CellsPerRoad; ++iLayer){
			pseudoTrackletsNumber=primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer];
			countCellsKernel.set_arg(0,primaryVertexContext.mGPUContext.boostPrimaryVertex);
			countCellsKernel.set_arg(1,primaryVertexContext.mGPUContext.boostLayerIndex[iLayer]);
			countCellsKernel.set_arg(2,primaryVertexContext.mGPUContext.boostTrackletsFoundForLayer);
			countCellsKernel.set_arg(3,primaryVertexContext.mGPUContext.boostTracklets[iLayer]);
			countCellsKernel.set_arg(4,primaryVertexContext.mGPUContext.boostTracklets[iLayer+1]);

			countCellsKernel.set_arg(5,primaryVertexContext.mGPUContext.boostClusters[iLayer]);
			countCellsKernel.set_arg(6,primaryVertexContext.mGPUContext.boostClusters[iLayer+1]);
			countCellsKernel.set_arg(7,primaryVertexContext.mGPUContext.boostClusters[iLayer+2]);
			countCellsKernel.set_arg(8, primaryVertexContext.mGPUContext.boostTrackletsLookupTable[iLayer]);
			if(iLayer==0)
				countCellsKernel.set_arg(9, boostFirstLayerCellsLookup);
			else
				countCellsKernel.set_arg(9, primaryVertexContext.mGPUContext.boostCellsLookupTable[iLayer-1]);


			if((pseudoTrackletsNumber % workgroupSize)!=0){
				int mult=pseudoTrackletsNumber/workgroupSize;
				pseudoTrackletsNumber=(mult+1)*workgroupSize;
			}
			try{
				boostQueue.enqueue_1d_range_kernel(countCellsKernel,0,pseudoTrackletsNumber,0);

			}catch(boost::compute::opencl_error &e){
						std::cout << e.error_string() << std::endl;
			}catch (std::exception& e) {
				std::cout<<e.what()<<std::endl;
			}
		}

		boostQueue.finish();

		//scan
		for (int iLayer { 0 }; iLayer<Constants::ITS::CellsPerRoad; ++iLayer) {
			iTrackletsNum=primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer];

			if(iLayer==0){

				// sort data on the device
				compute::inclusive_scan(
						boostFirstLayerCellsLookup.begin(),
						boostFirstLayerCellsLookup.end(),
						boostFirstLayerCellsLookup.begin(),
						boostQueue);

				// copy data back to the host
				compute::copy(
						boostFirstLayerCellsLookup.begin(),
						boostFirstLayerCellsLookup.end(),
						firstLayerLookUpTable,
						boostQueue);

				cellsFound[iLayer]=firstLayerLookUpTable[iTrackletsNum-1];
			}
			else{
				// sort data on the device
				compute::inclusive_scan(
						primaryVertexContext.mGPUContext.boostCellsLookupTable[iLayer-1].begin(),
						primaryVertexContext.mGPUContext.boostCellsLookupTable[iLayer-1].end(),
						primaryVertexContext.mGPUContext.boostCellsLookupTable[iLayer-1].begin(),
						boostQueue);
				// copy data back to the host
				compute::copy(
						primaryVertexContext.mGPUContext.boostCellsLookupTable[iLayer-1].begin(),
						primaryVertexContext.mGPUContext.boostCellsLookupTable[iLayer-1].end(),
						&(primaryVertexContext.mCellsLookupTable[iLayer-1][0]),
						boostQueue);

				cellsFound[iLayer]=primaryVertexContext.mCellsLookupTable[iLayer-1][iTrackletsNum-1];
			}

		}


		for (int iLayer{ 0 }; iLayer<Constants::ITS::CellsPerRoad; ++iLayer) {
			pseudoTrackletsNumber=primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer];
			computeCellsKernel.set_arg(0,primaryVertexContext.mGPUContext.boostPrimaryVertex);
			computeCellsKernel.set_arg(1,primaryVertexContext.mGPUContext.boostLayerIndex[iLayer]);
			computeCellsKernel.set_arg(2,primaryVertexContext.mGPUContext.boostTrackletsFoundForLayer);
			computeCellsKernel.set_arg(3,primaryVertexContext.mGPUContext.boostTracklets[iLayer]);
			computeCellsKernel.set_arg(4,primaryVertexContext.mGPUContext.boostTracklets[iLayer+1]);
			computeCellsKernel.set_arg(5,primaryVertexContext.mGPUContext.boostClusters[iLayer]);
			computeCellsKernel.set_arg(6,primaryVertexContext.mGPUContext.boostClusters[iLayer+1]);
			computeCellsKernel.set_arg(7,primaryVertexContext.mGPUContext.boostClusters[iLayer+2]);
			computeCellsKernel.set_arg(8, primaryVertexContext.mGPUContext.boostTrackletsLookupTable[iLayer]);
			if(iLayer==0)
				computeCellsKernel.set_arg(9, boostFirstLayerCellsLookup);
			else
				computeCellsKernel.set_arg(9, primaryVertexContext.mGPUContext.boostCellsLookupTable[iLayer-1]);
			computeCellsKernel.set_arg(10, primaryVertexContext.mGPUContext.boostCells[iLayer]);	//10

			if((pseudoTrackletsNumber % workgroupSize)!=0){
				int mult=pseudoTrackletsNumber/workgroupSize;
				pseudoTrackletsNumber=(mult+1)*workgroupSize;
			}

			try{
				boostQueue.enqueue_1d_range_kernel(computeCellsKernel,0,pseudoTrackletsNumber,0);

			}catch(boost::compute::opencl_error &e){
						std::cout << e.error_string() << std::endl;
			}catch (std::exception& e) {
				std::cout<<e.what()<<std::endl;
			}

		}

		boostQueue.finish();

		for (int iLayer{ 0 }; iLayer<Constants::ITS::CellsPerRoad; ++iLayer){
			primaryVertexContext.mCells[iLayer].resize(cellsFound[iLayer]);
			compute::copy(
					primaryVertexContext.mGPUContext.boostCells[iLayer].begin(),
					primaryVertexContext.mGPUContext.boostCells[iLayer].end(),
					primaryVertexContext.mCells[iLayer].begin(),
					boostQueue);


			if(iLayer>0){
				std::copy(primaryVertexContext.mCellsLookupTable[iLayer-1].begin(),
						primaryVertexContext.mCellsLookupTable[iLayer-1].end().operator --(),
						primaryVertexContext.mCellsLookupTable[iLayer-1].begin().operator ++());
				primaryVertexContext.mCellsLookupTable[iLayer-1][0]=0;
			}
		}

		free(firstLayerLookUpTable);
	}catch (std::exception& e) {
			std::cout<<e.what()<<std::endl;
			throw std::runtime_error { e.what() };

	}catch (...) {
		std::cout<<"Exception during compute tracklet phase"<<std::endl;
		throw std::runtime_error { "Exception during compute tracklet phase" };
	}
}




}
}
}
