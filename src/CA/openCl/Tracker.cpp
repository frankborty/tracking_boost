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
#include "ITSReconstruction/CA/openCl/Utils.h"
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
	int* trackletsFound;
	int totalTrackletsFound=0;
	cl::Buffer bTrackletLookUpTable;
	cl::Kernel oclCountTrackletKernel=GPU::Context::getInstance().getDeviceProperties().oclCountTrackletKernel;
	cl::Kernel oclComputeTrackletKernel=GPU::Context::getInstance().getDeviceProperties().oclComputeTrackletKernel;
	int workgroupSize=5*32;	//tmp value
	time_t tx,ty;

	try{
		cl::Context oclContext=GPU::Context::getInstance().getDeviceProperties().oclContext;
		cl::Device oclDevice=GPU::Context::getInstance().getDeviceProperties().oclDevice;
		cl::CommandQueue oclCommandQueue=GPU::Context::getInstance().getDeviceProperties().oclQueue;

	//boost_compute
		compute::device device = compute::device(oclDevice(),true);
		compute::context ctx(oclContext(),true);
		compute::command_queue queue(ctx, device);
	///


		iClustersNum=primaryVertexContext.mGPUContext.iClusterSize[0];

		if((iClustersNum % workgroupSize)!=0){
			int mult=iClustersNum/workgroupSize;
			iClustersNum=(mult+1)*workgroupSize;
		}


		firstLayerLookUpTable=(int*)malloc(iClustersNum*sizeof(int));
		memset(firstLayerLookUpTable,-1,iClustersNum*sizeof(int));
		bTrackletLookUpTable = cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				iClustersNum*sizeof(int),
				(void *) &firstLayerLookUpTable[0]);


		tx=clock();
		for (int iLayer{ 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer) {
			iClustersNum=primaryVertexContext.mGPUContext.iClusterSize[iLayer];
			oclCountTrackletKernel.setArg(0, primaryVertexContext.mGPUContext.bPrimaryVertex);
			oclCountTrackletKernel.setArg(1, primaryVertexContext.mGPUContext.bClusters[iLayer]);
			oclCountTrackletKernel.setArg(2, primaryVertexContext.mGPUContext.bClusters[iLayer+1]);
			oclCountTrackletKernel.setArg(3, primaryVertexContext.mGPUContext.bIndexTables[iLayer]);
			oclCountTrackletKernel.setArg(4, primaryVertexContext.mGPUContext.bLayerIndex[iLayer]);
			oclCountTrackletKernel.setArg(5, primaryVertexContext.mGPUContext.bTrackletsFoundForLayer);
			oclCountTrackletKernel.setArg(6, primaryVertexContext.mGPUContext.bClustersSize);
			if(iLayer==0)
				oclCountTrackletKernel.setArg(7, bTrackletLookUpTable);
			else
				oclCountTrackletKernel.setArg(7, primaryVertexContext.mGPUContext.bTrackletsLookupTable[iLayer-1]);

			int pseudoClusterNumber=iClustersNum;
			if((iClustersNum % workgroupSize)!=0){
				int mult=iClustersNum/workgroupSize;
				pseudoClusterNumber=(mult+1)*workgroupSize;
			}

//			time_t tx=clock();
			GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].enqueueNDRangeKernel(
				oclCountTrackletKernel,
				cl::NullRange,
				cl::NDRange(pseudoClusterNumber),
				cl::NDRange(workgroupSize));
				//cl::NullRange);
/*
			GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].finish();

			trackletsFound = (int *) GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].enqueueMapBuffer(
					primaryVertexContext.mGPUContext.bTrackletsFoundForLayer,
					CL_TRUE, // block
					CL_MAP_READ,
					0,
					Constants::ITS::TrackletsPerRoad*sizeof(int)
			);

			totalTrackletsFound+=trackletsFound[iLayer];
			//std::cout<<"Tracklets found for layer "<<iLayer<<": "<<trackletsFound[iLayer]<<std::endl;
*/

		}
		free(firstLayerLookUpTable);
		//std::cout<<"Total tracklets found: "<<totalTrackletsFound<<std::endl;


	ty=clock();
	float time = ((float) ty - (float) tx) / (CLOCKS_PER_SEC / 1000);
	std::cout<< "\t>Total tracklet count time = "<<time<<" ms" << std::endl;

	//scan
	tx=clock();
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
			compute::exclusive_scan(device_vector.begin(),device_vector.end(),device_vector.begin(),0,queue);
			// copy data back to the host
			compute::copy(device_vector.begin(), device_vector.end(), lookUpFound, queue);
			bTrackletLookUpTable=cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				iClustersNum*sizeof(int),
				(void *) lookUpFound);


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
			compute::exclusive_scan(device_vector.begin(),device_vector.end(),device_vector.begin(),0,queue);
			// copy data back to the host
			compute::copy(device_vector.begin(), device_vector.end(), lookUpFound, queue);

			primaryVertexContext.mGPUContext.bTrackletsLookupTable[iLayer-1]=cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				iClustersNum*sizeof(int),
				(void *) lookUpFound);



		}
	}
	ty=clock();
	time = ((float) ty - (float) tx) / (CLOCKS_PER_SEC / 1000);
	std::cout<< "\t>Total scan time = "<<time<<" ms" << std::endl;

	trackletsFound = (int *) oclCommandQueue.enqueueMapBuffer(
			primaryVertexContext.mGPUContext.bTrackletsFoundForLayer,
			CL_TRUE, // block
			CL_MAP_READ,
			0,
			Constants::ITS::TrackletsPerRoad*sizeof(int)
	);
	for(int i=0;i<Constants::ITS::TrackletsPerRoad;i++){
		primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[i]=trackletsFound[i];
	}

	//calcolo le tracklet
	tx=clock();
	for (int iLayer{ 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer) {
		iClustersNum=primaryVertexContext.mGPUContext.iClusterSize[iLayer];
		oclComputeTrackletKernel.setArg(0, primaryVertexContext.mGPUContext.bPrimaryVertex);
		oclComputeTrackletKernel.setArg(1, primaryVertexContext.mGPUContext.bClusters[iLayer]);
		oclComputeTrackletKernel.setArg(2, primaryVertexContext.mGPUContext.bClusters[iLayer+1]);
		oclComputeTrackletKernel.setArg(3, primaryVertexContext.mGPUContext.bIndexTables[iLayer]);
		oclComputeTrackletKernel.setArg(4, primaryVertexContext.mGPUContext.bTracklets[iLayer]);
		oclComputeTrackletKernel.setArg(5, primaryVertexContext.mGPUContext.bLayerIndex[iLayer]);
		oclComputeTrackletKernel.setArg(6, primaryVertexContext.mGPUContext.bTrackletsFoundForLayer);
		oclComputeTrackletKernel.setArg(7, primaryVertexContext.mGPUContext.bClustersSize);
		if(iLayer==0)
			oclComputeTrackletKernel.setArg(8, bTrackletLookUpTable);
		else
			oclComputeTrackletKernel.setArg(8, primaryVertexContext.mGPUContext.bTrackletsLookupTable[iLayer-1]);

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


	ty=clock();
	time = ((float) ty - (float) tx) / (CLOCKS_PER_SEC / 1000);
	std::cout<< "\t>Total compute tracklet time = "<<time<<" ms" << std::endl;



	}catch (...) {
		std::cout<<"Exception during compute tracklet phase"<<std::endl;
		throw std::runtime_error { "Exception during compute cells phase" };
	}
}

template<>
void TrackerTraits<true>::computeLayerCells(CA::PrimaryVertexContext& primaryVertexContext)
{

	time_t tx,ty;
	int iTrackletsNum;
	int *firstLayerLookUpTable;
	int* trackletsFound;
	int totalCellsFound=0;
	int *cellsFound;
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

		tx=clock();
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
			oclCountCellKernel.setArg(10, primaryVertexContext.mGPUContext.bCellsFoundForLayer);


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

			GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].finish();
		}

		delete []firstLayerLookUpTable;
		ty=clock();
		float time = ((float) ty - (float) tx) / (CLOCKS_PER_SEC / 1000);
		std::cout<< "\t>Total cells count time = "<<time<<" ms" << std::endl;

		//scan
		tx=clock();
		for (int iLayer { 0 }; iLayer<Constants::ITS::CellsPerRoad; ++iLayer) {
			iTrackletsNum=primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer];
			GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].finish();

			cellsFound = (int *) GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].enqueueMapBuffer(
					primaryVertexContext.mGPUContext.bCellsFoundForLayer,
					CL_TRUE, // block
					CL_MAP_READ,
					0,
					5*sizeof(int)
			);


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
				compute::exclusive_scan(device_vector.begin(),device_vector.end(),device_vector.begin(),0,queue);
				// copy data back to the host
				compute::copy(device_vector.begin(), device_vector.end(), lookUpFound, queue);
				bCellLookUpTable=cl::Buffer(
					oclContext,
					(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
					iTrackletsNum*sizeof(int),
					(void *) lookUpFound);


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
				compute::exclusive_scan(device_vector.begin(),device_vector.end(),device_vector.begin(),0,queue);
				// copy data back to the host
				compute::copy(device_vector.begin(), device_vector.end(), lookUpFound, queue);

				primaryVertexContext.mGPUContext.bCellsLookupTable[iLayer-1]=cl::Buffer(
					oclContext,
					(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
					iTrackletsNum*sizeof(int),
					(void *) lookUpFound);
			}
		}
		ty=clock();
		time = ((float) ty - (float) tx) / (CLOCKS_PER_SEC / 1000);
		std::cout<< "\t>Total cells scan time = "<<time<<" ms" << std::endl;

		return;

		//compute cells
		//calcolo le tracklet
		//std::cout<<"calcolo le tracklet"<<std::endl;
		tx=clock();
		for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad;++iLayer) {
			GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].finish();
			oclComputeCellKernel.setArg(0, primaryVertexContext.mGPUContext.bPrimaryVertex);  //0 fPrimaryVertex
			oclComputeCellKernel.setArg(1, primaryVertexContext.mGPUContext.bLayerIndex[iLayer]); //1 iCurrentLayer
			oclComputeCellKernel.setArg(2, primaryVertexContext.mGPUContext.bTrackletsFoundForLayer);  //2 iLayerTrackletSize
			oclComputeCellKernel.setArg(3, primaryVertexContext.mGPUContext.bTracklets[iLayer]); //3  currentLayerTracklets
			oclComputeCellKernel.setArg(4, primaryVertexContext.mGPUContext.bTracklets[iLayer+1]); //4 nextLayerTracklets
			oclComputeCellKernel.setArg(5, primaryVertexContext.mGPUContext.bClusters[iLayer]);  //6 currentLayerClusters
			oclComputeCellKernel.setArg(6, primaryVertexContext.mGPUContext.bClusters[iLayer+1]);//7 nextLayerClusters
			oclComputeCellKernel.setArg(7, primaryVertexContext.mGPUContext.bClusters[iLayer+2]);//8 next2LayerClusters
			oclComputeCellKernel.setArg(8, primaryVertexContext.mGPUContext.bTrackletsLookupTable[iLayer]);//9  currentLayerTrackletsLookupTable

			if(iLayer==0)
				oclComputeCellKernel.setArg(9, bCellLookUpTable);//9iCellsPerTrackletPreviousLayer;
			else
				oclComputeCellKernel.setArg(9, primaryVertexContext.mGPUContext.bCellsLookupTable[iLayer-1]);//9iCellsPerTrackletPreviousLayer
			oclComputeCellKernel.setArg(10, primaryVertexContext.mGPUContext.bCells[iLayer]);

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

			GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].finish();

/*
			CellStruct* output = (CellStruct *) oclCommandqueues[iLayer].enqueueMapBuffer(
				primaryVertexContext.openClPrimaryVertexContext.bCells[iLayer],
				CL_TRUE, // block
				CL_MAP_READ,
				0,
				cellsFound[iLayer]*sizeof(CellStruct)
			);
			oclCommandqueues[iLayer].finish();
			outFileCell<<"Cell found starting from layer #"<<iLayer<<"	total:"<<cellsFound[iLayer]<<"\n";
			for(int i=0;i<cellsFound[iLayer];i++){
				//std::cout<<i<<std::endl;
				outFileCell<<output[i].mFirstTrackletIndex<<"\t"<<output[i].mSecondTrackletIndex<<"\t"<<output[i].mCurvature<<"\t"<<output[i].mLevel<<"\t"<<output[i].mFirstClusterIndex<<"\t"<<output[i].mSecondClusterIndex<<"\t"<<output[i].mThirdClusterIndex<<"\n";
				//outFileCell<<output[i].mFirstClusterIndex<<"\n";
			}
			std::cout<<"end layer "<<iLayer<<std::endl;
*/
		}
		ty=clock();
		time = ((float) ty - (float) tx) / (CLOCKS_PER_SEC / 1000);
		std::cout<< "\t>Total cells compute time = "<<time<<" ms" << std::endl;






	}catch(const cl::Error &err){
		std::string errString=o2::ITS::CA::GPU::Utils::OCLErr_code(err.err());
		//std::cout<< errString << std::endl;
		throw std::runtime_error { errString };
	}
	catch( const std::exception & ex ) {
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
