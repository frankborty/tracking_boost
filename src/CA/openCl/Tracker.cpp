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
#include "boost/compute.hpp"
namespace compute = boost::compute;

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
	int *firstLayerLookUpTable;
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
	///


		iClustersNum=primaryVertexContext.mClusters[0].size();

		if((iClustersNum % workgroupSize)!=0){
			int mult=iClustersNum/workgroupSize;
			iClustersNum=(mult+1)*workgroupSize;
		}


		firstLayerLookUpTable=(int*)malloc(iClustersNum*sizeof(int));
		memset(firstLayerLookUpTable,-1,iClustersNum*sizeof(int));
		boostTrackletLookUpTable = compute::buffer(
				boostContext,
				iClustersNum*sizeof(int),
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				(void *) &firstLayerLookUpTable[0]);


		for (int iLayer{ 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer){
			countTrackletKernel.set_arg(0,primaryVertexContext.mGPUContext.boostPrimaryVertex);
			countTrackletKernel.set_arg(1,primaryVertexContext.mGPUContext.boostClusters[iLayer]);
			countTrackletKernel.set_arg(2,primaryVertexContext.mGPUContext.boostClusters[iLayer+1]);
			countTrackletKernel.set_arg(3,primaryVertexContext.mGPUContext.boostIndexTables[iLayer]);
			countTrackletKernel.set_arg(4,primaryVertexContext.mGPUContext.boostLayerIndex[iLayer]);
			countTrackletKernel.set_arg(5,primaryVertexContext.mGPUContext.boostClusterSize);
			if(iLayer==0)
				countTrackletKernel.set_arg(6, boostTrackletLookUpTable);
			else
				countTrackletKernel.set_arg(6, primaryVertexContext.mGPUContext.boostTrackletsLookupTable[iLayer-1]);


			boostQueue.enqueue_1d_range_kernel(countTrackletKernel,0,1,1);
			boostQueue.finish();
		}

/*
		for (int iLayer{ 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer) {
			iClustersNum=primaryVertexContext.mGPUContext.iClusterSize[iLayer];
			oclCountTrackletKernel.setArg(0, primaryVertexContext.mGPUContext.bPrimaryVertex);
			oclCountTrackletKernel.setArg(1, primaryVertexContext.mGPUContext.bClusters[iLayer]);
			oclCountTrackletKernel.setArg(2, primaryVertexContext.mGPUContext.bClusters[iLayer+1]);
			oclCountTrackletKernel.setArg(3, primaryVertexContext.mGPUContext.bIndexTables[iLayer]);
			oclCountTrackletKernel.setArg(4, primaryVertexContext.mGPUContext.bLayerIndex[iLayer]);
			//oclCountTrackletKernel.setArg(5, primaryVertexContext.mGPUContext.bTrackletsFoundForLayer);
			oclCountTrackletKernel.setArg(5, primaryVertexContext.mGPUContext.bClustersSize);
			if(iLayer==0)
				oclCountTrackletKernel.setArg(6, bTrackletLookUpTable);
			else
				oclCountTrackletKernel.setArg(6, primaryVertexContext.mGPUContext.bTrackletsLookupTable[iLayer-1]);

			int pseudoClusterNumber=iClustersNum;
			if((iClustersNum % workgroupSize)!=0){
				int mult=iClustersNum/workgroupSize;
				pseudoClusterNumber=(mult+1)*workgroupSize;
			}


			GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].enqueueNDRangeKernel(
				oclCountTrackletKernel,
				cl::NullRange,
				cl::NDRange(pseudoClusterNumber),
				cl::NDRange(workgroupSize));

		}
		free(firstLayerLookUpTable);

	//scan
	for (int iLayer { 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer) {
		iClustersNum=primaryVertexContext.mGPUContext.iClusterSize[iLayer];
		GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].finish();

		if(iLayer==0){
			int* lookUpFound = (int *) GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].enqueueMapBuffer(
				bTrackletLookUpTable,
				CL_TRUE, // block
				CL_MAP_READ,
				0,
				iClustersNum*sizeof(int)
			);

			// create vector on the device
			compute::vector<int> device_vector(iClustersNum, ctx);

			// copy data to the device
			compute::copy(lookUpFound, lookUpFound+iClustersNum, device_vector.begin(), queue);

			// sort data on the device
			compute::inclusive_scan(device_vector.begin(),device_vector.end(),device_vector.begin(),queue);
			// copy data back to the host
			compute::copy(device_vector.begin(), device_vector.end(), lookUpFound, queue);
			bTrackletLookUpTable=cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				iClustersNum*sizeof(int),
				(void *) lookUpFound);
			//for(int j=0;j<iClustersNum;j++)
				//std::cout<<"["<<j<<"]: "<<lookUpFound[j]<<std::endl;
			primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer]=lookUpFound[iClustersNum-1];

		}
		else{

			int* lookUpFound = (int *) GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].enqueueMapBuffer(
				primaryVertexContext.mGPUContext.bTrackletsLookupTable[iLayer-1],
				CL_TRUE, // block
				CL_MAP_READ,
				0,
				iClustersNum*sizeof(int)
			);

			// create vector on the device
			compute::vector<int> device_vector(iClustersNum, ctx);

			// copy data to the device
			compute::copy(lookUpFound, lookUpFound+iClustersNum, device_vector.begin(), queue);

			// sort data on the device
			compute::inclusive_scan(device_vector.begin(),device_vector.end(),device_vector.begin(),queue);
			// copy data back to the host
			compute::copy(device_vector.begin(), device_vector.end(), lookUpFound, queue);

			primaryVertexContext.mGPUContext.bTrackletsLookupTable[iLayer-1]=cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				iClustersNum*sizeof(int),
				(void *) lookUpFound);
			//for(int j=0;j<iClustersNum;j++)
				//std::cout<<"["<<j<<"]: "<<lookUpFound[j]<<std::endl;
			primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer]=lookUpFound[iClustersNum-1];

		}
	}

	primaryVertexContext.mGPUContext.bTrackletsFoundForLayer=cl::Buffer(
			oclContext,
			(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			o2::ITS::CA::Constants::ITS::TrackletsPerRoad*sizeof(int),
			(void *) primaryVertexContext.mGPUContext.iTrackletFoundPerLayer);



	//calcolo le tracklet
	for (int iLayer{ 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer) {
		iClustersNum=primaryVertexContext.mGPUContext.iClusterSize[iLayer];
		oclComputeTrackletKernel.setArg(0, primaryVertexContext.mGPUContext.bPrimaryVertex);
		oclComputeTrackletKernel.setArg(1, primaryVertexContext.mGPUContext.bClusters[iLayer]);
		oclComputeTrackletKernel.setArg(2, primaryVertexContext.mGPUContext.bClusters[iLayer+1]);
		oclComputeTrackletKernel.setArg(3, primaryVertexContext.mGPUContext.bIndexTables[iLayer]);
		oclComputeTrackletKernel.setArg(4, primaryVertexContext.mGPUContext.bTracklets[iLayer]);
		oclComputeTrackletKernel.setArg(5, primaryVertexContext.mGPUContext.bLayerIndex[iLayer]);
		//oclComputeTrackletKernel.setArg(6, primaryVertexContext.mGPUContext.bTrackletsFoundForLayer);
		oclComputeTrackletKernel.setArg(6, primaryVertexContext.mGPUContext.bClustersSize);
		if(iLayer==0)
			oclComputeTrackletKernel.setArg(7, bTrackletLookUpTable);
		else
			oclComputeTrackletKernel.setArg(7, primaryVertexContext.mGPUContext.bTrackletsLookupTable[iLayer-1]);

		int pseudoClusterNumber=iClustersNum;
		if((iClustersNum % workgroupSize)!=0){
			int mult=iClustersNum/workgroupSize;
			pseudoClusterNumber=(mult+1)*workgroupSize;
		}



		GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].enqueueNDRangeKernel(
			oclComputeTrackletKernel,
			cl::NullRange,
			cl::NDRange(pseudoClusterNumber),
			cl::NDRange(workgroupSize));


	}

	for (int iLayer{ 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer)
		GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].finish();
*/
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
	int *firstLayerLookUpTable;
	int cellsFound[Constants::ITS::CellsPerRoad];
	cl::Buffer bCellLookUpTable;
	cl::Kernel oclCountCellKernel=GPU::Context::getInstance().getDeviceProperties().oclCountCellKernel;
	cl::Kernel oclComputeCellKernel=GPU::Context::getInstance().getDeviceProperties().oclComputeCellKernel;
	int workgroupSize=5*32;
	try{
		cl::Context oclContext=GPU::Context::getInstance().getDeviceProperties().oclContext;
		cl::Device oclDevice=GPU::Context::getInstance().getDeviceProperties().oclDevice;
		cl::CommandQueue oclCommandQueue=GPU::Context::getInstance().getDeviceProperties().oclQueue;
		//boost_compute
			compute::device device = compute::device(oclDevice(),true);
			compute::context ctx(oclContext(),true);
			compute::command_queue queue(ctx, device);
		///
		iTrackletsNum=primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[0];

		int pseudoTracletsNumber=iTrackletsNum;
		if((pseudoTracletsNumber % workgroupSize)!=0){
			int mult=pseudoTracletsNumber/workgroupSize;
			pseudoTracletsNumber=(mult+1)*workgroupSize;
		}
		firstLayerLookUpTable = new int[pseudoTracletsNumber];

		//std::fill(firstLayerLookUpTable,firstLayerLookUpTable+pseudoTracletsNumber,-1);
		memset(firstLayerLookUpTable,-1,pseudoTracletsNumber*sizeof(int));
		bCellLookUpTable = cl::Buffer(
			oclContext,
			(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			pseudoTracletsNumber*sizeof(int),
			(void *) &firstLayerLookUpTable[0]);

		for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad;++iLayer) {
			GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].finish();
			oclCountCellKernel.setArg(0, primaryVertexContext.mGPUContext.bPrimaryVertex);  //0 fPrimaryVertex
			oclCountCellKernel.setArg(1, primaryVertexContext.mGPUContext.bLayerIndex[iLayer]); //1 iCurrentLayer
			oclCountCellKernel.setArg(2, primaryVertexContext.mGPUContext.bTrackletsFoundForLayer);  //2 iLayerTrackletSize
			oclCountCellKernel.setArg(3, primaryVertexContext.mGPUContext.bTracklets[iLayer]); //3  currentLayerTracklets
			oclCountCellKernel.setArg(4, primaryVertexContext.mGPUContext.bTracklets[iLayer+1]); //4 nextLayerTracklets				oclCountCellKernel.setArg(5, primaryVertexContext.mGPUContext.bTracklets[iLayer+2]); //5 next2LayerTracklets
			oclCountCellKernel.setArg(5, primaryVertexContext.mGPUContext.bClusters[iLayer]);  //5 currentLayerClusters
			oclCountCellKernel.setArg(6, primaryVertexContext.mGPUContext.bClusters[iLayer+1]);//6 nextLayerClusters
			oclCountCellKernel.setArg(7, primaryVertexContext.mGPUContext.bClusters[iLayer+2]);//7 next2LayerClusters
			oclCountCellKernel.setArg(8, primaryVertexContext.mGPUContext.bTrackletsLookupTable[iLayer]);//8  currentLayerTrackletsLookupTable

			if(iLayer==0)
				oclCountCellKernel.setArg(9, bCellLookUpTable);//9iCellsPerTrackletPreviousLayer;
			else
				oclCountCellKernel.setArg(9, primaryVertexContext.mGPUContext.bCellsLookupTable[iLayer-1]);//9iCellsPerTrackletPreviousLayer
			//oclCountCellKernel.setArg(10, primaryVertexContext.mGPUContext.bCellsFoundForLayer);


			int pseudoTrackletsNumber=primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer];
			if((pseudoTrackletsNumber % workgroupSize)!=0){
				int mult=pseudoTrackletsNumber/workgroupSize;
				pseudoTrackletsNumber=(mult+1)*workgroupSize;
			}

			GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].enqueueNDRangeKernel(
				oclCountCellKernel,
				cl::NullRange,
				cl::NDRange(pseudoTrackletsNumber),
				cl::NDRange(workgroupSize));


		}

		delete []firstLayerLookUpTable;

		for (int iLayer { 0 }; iLayer<Constants::ITS::CellsPerRoad; ++iLayer)
			GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].finish();


		//scan
		for (int iLayer { 0 }; iLayer<Constants::ITS::CellsPerRoad; ++iLayer) {
			iTrackletsNum=primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer];
			//GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].finish();

			if(iLayer==0){
				int* lookUpFound = (int *) GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].enqueueMapBuffer(
					bCellLookUpTable,
					CL_TRUE, // block
					CL_MAP_READ,
					0,
					iTrackletsNum*sizeof(int)
				);

				// create vector on the device
				compute::vector<int> device_vector(iTrackletsNum, ctx);

				// copy data to the device
				compute::copy(lookUpFound, lookUpFound+iTrackletsNum, device_vector.begin(), queue);

				// sort data on the device
				compute::inclusive_scan(device_vector.begin(),device_vector.end(),device_vector.begin(),queue);
				// copy data back to the host
				compute::copy(device_vector.begin(), device_vector.end(), lookUpFound, queue);
				bCellLookUpTable=cl::Buffer(
					oclContext,
					(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
					iTrackletsNum*sizeof(int),
					(void *) lookUpFound);
				//std::cout<<"["<<iLayer<<"]: "<<lookUpFound[iTrackletsNum-1]<<std::endl;
				cellsFound[iLayer]=lookUpFound[iTrackletsNum-1];
			}
			else{

				int* lookUpFound = (int *) GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].enqueueMapBuffer(
					primaryVertexContext.mGPUContext.bCellsLookupTable[iLayer-1],
					CL_TRUE, // block
					CL_MAP_READ,
					0,
					iTrackletsNum*sizeof(int)
				);

				// create vector on the device
				compute::vector<int> device_vector(iTrackletsNum, ctx);

				// copy data to the device
				compute::copy(lookUpFound, lookUpFound+iTrackletsNum, device_vector.begin(), queue);

				// sort data on the device
				compute::inclusive_scan(device_vector.begin(),device_vector.end(),device_vector.begin(),queue);
				// copy data back to the host
				compute::copy(device_vector.begin(), device_vector.end(), lookUpFound, queue);

				primaryVertexContext.mGPUContext.bCellsLookupTable[iLayer-1]=cl::Buffer(
					oclContext,
					(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
					iTrackletsNum*sizeof(int),
					(void *) lookUpFound);
				//std::cout<<"["<<iLayer<<"]: "<<lookUpFound[iTrackletsNum-1]<<std::endl;
				cellsFound[iLayer]=lookUpFound[iTrackletsNum-1];

			}
		}



		//compute cells
		//std::cout<<"calcolo le cells"<<std::endl;
		for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad;++iLayer) {
			//std::cout<<"start layer "<<iLayer<<std::endl;
			GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].finish();
			oclComputeCellKernel.setArg(0, primaryVertexContext.mGPUContext.bPrimaryVertex);  //0 fPrimaryVertex
			oclComputeCellKernel.setArg(1, primaryVertexContext.mGPUContext.bLayerIndex[iLayer]); //1 iCurrentLayer
			oclComputeCellKernel.setArg(2, primaryVertexContext.mGPUContext.bTrackletsFoundForLayer);  //2 iLayerTrackletSize
			oclComputeCellKernel.setArg(3, primaryVertexContext.mGPUContext.bTracklets[iLayer]); //3  currentLayerTracklets
			oclComputeCellKernel.setArg(4, primaryVertexContext.mGPUContext.bTracklets[iLayer+1]); //4 nextLayerTracklets
			oclComputeCellKernel.setArg(5, primaryVertexContext.mGPUContext.bClusters[iLayer]);  //5 currentLayerClusters
			oclComputeCellKernel.setArg(6, primaryVertexContext.mGPUContext.bClusters[iLayer+1]);//6 nextLayerClusters
			oclComputeCellKernel.setArg(7, primaryVertexContext.mGPUContext.bClusters[iLayer+2]);//7 next2LayerClusters
			oclComputeCellKernel.setArg(8, primaryVertexContext.mGPUContext.bTrackletsLookupTable[iLayer]);//8  currentLayerTrackletsLookupTable

			if(iLayer==0)
				oclComputeCellKernel.setArg(9, bCellLookUpTable);//9iCellsPerTrackletPreviousLayer;
			else
				oclComputeCellKernel.setArg(9, primaryVertexContext.mGPUContext.bCellsLookupTable[iLayer-1]);//9 iCellsPerTrackletPreviousLayer
			oclComputeCellKernel.setArg(10, primaryVertexContext.mGPUContext.bCells[iLayer]);	//10

			int pseudoTrackersNumber=primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer];
			if((pseudoTrackersNumber % workgroupSize)!=0){
				int mult=pseudoTrackersNumber/workgroupSize;
				pseudoTrackersNumber=(mult+1)*workgroupSize;
			}
			GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].enqueueNDRangeKernel(
				oclComputeCellKernel,
				cl::NullRange,
				cl::NDRange(pseudoTrackersNumber),
				cl::NDRange(workgroupSize));


		}

		for(int iLayer=0;iLayer<Constants::ITS::CellsPerRoad;iLayer++){
			GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].finish();
			CellStruct* mCells = (CellStruct *) GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].enqueueMapBuffer(
				primaryVertexContext.mGPUContext.bCells[iLayer],
				CL_TRUE, // block
				CL_MAP_READ,
				0,
				cellsFound[iLayer]*sizeof(CellStruct)
			);

			for(int j {0};j<cellsFound[iLayer];j++){
				primaryVertexContext.getCells()[iLayer].emplace_back(mCells[j]);
			}
			if(iLayer>0){
				int trackletsNum=primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer];
				int* lookUpFound = (int *) GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].enqueueMapBuffer(
						primaryVertexContext.mGPUContext.bCellsLookupTable[iLayer-1],
						CL_TRUE, // block
						CL_MAP_READ,
						0,
						trackletsNum*sizeof(int)
				);
				if(iLayer>=1){
					for(int j {1};j<trackletsNum;j++){
						if(lookUpFound[j]!=0){
							primaryVertexContext.mCellsLookupTable[iLayer-1][j]=lookUpFound[j-1];
						}
						primaryVertexContext.mCellsLookupTable[iLayer-1][0]=0;
					}

				}
			}
		}


	}catch(const cl::Error &err){
		std::string errString=o2::ITS::CA::GPU::Utils::OCLErr_code(err.err());
		std::cout << "Allocation failed: " << err.what() << '\n';
		throw std::runtime_error { errString };
	}
	catch( const std::exception & ex ) {
		std::cout << "Allocation failed: " << ex.what() << '\n';
	       throw std::runtime_error { ex.what() };
	}
  	catch (...) {
		std::cout<<"Exception during compute cells phase"<<std::endl;
		throw std::runtime_error { "Exception during compute cells phase" };
	}
}




}
}
}
