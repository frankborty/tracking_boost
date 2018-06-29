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

#if TRACKINGITSU_OCL_MODE
	BOOST_COMPUTE_TYPE_NAME(const int, int);
	BOOST_COMPUTE_TYPE_NAME(const float, float);
	BOOST_COMPUTE_ADAPT_STRUCT(o2::ITS::CA::Tracklet, Tracklet, ( firstClusterIndex, secondClusterIndex, tanLambda, phiCoordinate));
	//BOOST_COMPUTE_TYPE_NAME(compute::float_[], compute::float[]);
	BOOST_COMPUTE_ADAPT_STRUCT(o2::ITS::CA::Cell, Cell, ( mFirstClusterIndex, mSecondClusterIndex, mThirdClusterIndex,mFirstTrackletIndex,mSecondTrackletIndex,mNormalVectorCoordinates,mCurvature, mLevel));
#endif

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
	compute::command_queue boostQueues[Constants::ITS::TrackletsPerRoad];
	compute::buffer boostTrackletLookUpTable;
	int workgroupSize=5*32;
	try{


		//boost_compute
		compute::device boostDevice = GPU::Context::getInstance().getBoostDeviceProperties().boostDevice;
		compute::context boostContext= GPU::Context::getInstance().getBoostDeviceProperties().boostContext;
		compute::kernel computeTrackletKernel=GPU::Context::getInstance().getBoostDeviceProperties().computeTrackletsBoostKernel;


		for(int i=0;i<Constants::ITS::TrackletsPerRoad;i++){
			boostQueues[i]=GPU::Context::getInstance().getBoostDeviceProperties().boostCommandQueues[i];
			boostQueues[i].finish();
		}


		iClustersNum=primaryVertexContext.mClusters[0].size();
		if((iClustersNum % workgroupSize)!=0){
			int mult=iClustersNum/workgroupSize;
			iClustersNum=(mult+1)*workgroupSize;
		}

		int iTrackletFoundPerLayer[]={0,0,0,0,0,0};
		primaryVertexContext.mGPUContext.boostTrackletsFoundForLayer=compute::vector<int>(Constants::ITS::TrackletsPerRoad,boostContext);
		boost::compute::copy_n(
				iTrackletFoundPerLayer,
				6,
				primaryVertexContext.mGPUContext.boostTrackletsFoundForLayer.begin(),
				boostQueues[0]
		);

		int *firstLayerLookUpTable;
		firstLayerLookUpTable=(int*)malloc(iClustersNum*sizeof(int));
		memset(firstLayerLookUpTable,0,iClustersNum*sizeof(int));
		boost::compute::vector<int> boostFirstLayerTrackletsLookup(iClustersNum,boostContext);
		boost::compute::copy_n(
				firstLayerLookUpTable,
				iClustersNum,
				boostFirstLayerTrackletsLookup.begin(),
				boostQueues[0]
		);

		for (int iLayer{ 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer){
			iClustersNum=primaryVertexContext.mClusters[iLayer].size();
			computeTrackletKernel.set_arg(0,primaryVertexContext.mGPUContext.boostPrimaryVertex);
			computeTrackletKernel.set_arg(1,primaryVertexContext.mGPUContext.boostClusters[iLayer]);
			computeTrackletKernel.set_arg(2,primaryVertexContext.mGPUContext.boostClusters[iLayer+1]);
			computeTrackletKernel.set_arg(3,primaryVertexContext.mGPUContext.boostTracklets[iLayer]);
			computeTrackletKernel.set_arg(4,primaryVertexContext.mGPUContext.boostIndexTables[iLayer]);
			computeTrackletKernel.set_arg(5,primaryVertexContext.mGPUContext.boostLayerIndex[iLayer]);
			computeTrackletKernel.set_arg(6,primaryVertexContext.mGPUContext.boostClusterSize);
			if(iLayer==0)
				computeTrackletKernel.set_arg(7, boostFirstLayerTrackletsLookup);
			else
				computeTrackletKernel.set_arg(7, primaryVertexContext.mGPUContext.boostTrackletsLookupTable[iLayer-1]);
			computeTrackletKernel.set_arg(8,primaryVertexContext.mGPUContext.boostTrackletsFoundForLayer);

			int pseudoClusterNumber=iClustersNum;
			if((iClustersNum % workgroupSize)!=0){
				int mult=iClustersNum/workgroupSize;
				pseudoClusterNumber=(mult+1)*workgroupSize;
			}
			boostQueues[iLayer].enqueue_1d_range_kernel(computeTrackletKernel,0,pseudoClusterNumber,workgroupSize);
		}

		//scan
		for (int iLayer { 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer) {
			iClustersNum=primaryVertexContext.mGPUContext.iClusterSize[iLayer];
			boostQueues[iLayer].finish();

			if(iLayer==0){

				// sort data on the device
				compute::inclusive_scan(
						boostFirstLayerTrackletsLookup.begin(),
						boostFirstLayerTrackletsLookup.end(),
						boostFirstLayerTrackletsLookup.begin(),
						boostQueues[0]);

				// copy data back to the host
				compute::copy_n(
						boostFirstLayerTrackletsLookup.begin(),
						iClustersNum,
						firstLayerLookUpTable,
						boostQueues[0]);

				primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer]=firstLayerLookUpTable[iClustersNum-1];
				//std::cout<<"["<<iLayer<<"]: "<<primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer]<<std::endl;
			}
			else{
				// sort data on the device
				compute::inclusive_scan(
						primaryVertexContext.mGPUContext.boostTrackletsLookupTable[iLayer-1].begin(),
						primaryVertexContext.mGPUContext.boostTrackletsLookupTable[iLayer-1].end(),
						primaryVertexContext.mGPUContext.boostTrackletsLookupTable[iLayer-1].begin(),
						boostQueues[iLayer]);
				// copy data back to the host
				compute::copy_n(
						primaryVertexContext.mGPUContext.boostTrackletsLookupTable[iLayer-1].begin(),
						iClustersNum,
						primaryVertexContext.mTrackletsLookupTable[iLayer-1].begin(),
						boostQueues[iLayer]);

				primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer]=primaryVertexContext.mTrackletsLookupTable[iLayer-1][iClustersNum-1];
				//std::cout<<"["<<iLayer<<"]: "<<primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer]<<std::endl;
			}
		}


		BOOST_COMPUTE_FUNCTION(bool, sort_by_x, (Tracklet a, Tracklet b),
		{
			return a.firstClusterIndex < b.firstClusterIndex;
		});

		//std::cout<<"Track found"<<std::endl;
		for (int iLayer{ 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer) {
			int iTrackNumber=primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer];
			//std::cout<<"["<<iLayer<<"]: "<<iTrackNumber<<std::endl;
			compute::stable_sort(primaryVertexContext.mGPUContext.boostTracklets[iLayer].begin(),
					primaryVertexContext.mGPUContext.boostTracklets[iLayer].begin().operator +=(iTrackNumber),
					sort_by_x,
					boostQueues[iLayer]);
			/*if(iLayer==0){
				compute::copy_n(
					primaryVertexContext.mGPUContext.boostTracklets[iLayer].begin(),
					iTrackNumber,
					primaryVertexContext.mTracklets[iLayer].begin(),
					boostQueues[iLayer]);
				for(int i=0;i<iTrackNumber;i++){
					Tracklet t=primaryVertexContext.mTracklets[iLayer][i];
					std::cout<<t.firstClusterIndex<<"\t"<<t.secondClusterIndex<<"\t"
							 <<t.tanLambda<<"\t"<<t.phiCoordinate<<std::endl;
				}
			}*/
		}

		free(firstLayerLookUpTable);
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

template<>
void TrackerTraits<true>::computeLayerCells(CA::PrimaryVertexContext& primaryVertexContext)
{
	int iTrackletsNum;
	int workgroupSize=5*32;
	compute::command_queue boostQueues[Constants::ITS::TrackletsPerRoad];
	try{
		compute::context boostContext=GPU::Context::getInstance().getBoostDeviceProperties().boostContext;
		compute::kernel computeCellsBoostKernel=GPU::Context::getInstance().getBoostDeviceProperties().computeCellsBoostKernel;
		for(int i=0;i<Constants::ITS::TrackletsPerRoad;i++){
			boostQueues[i]=GPU::Context::getInstance().getBoostDeviceProperties().boostCommandQueues[i];
			boostQueues[i].finish();
		}



		boost::compute::vector<int> tmpboostCellsFoundForLayer;
		int iCellFoundPerLayer[]={0,0,0,0,0};
		primaryVertexContext.mGPUContext.boostCellsFoundForLayer=compute::vector<int>(Constants::ITS::CellsPerRoad,boostContext);
		boost::compute::copy(
				iCellFoundPerLayer,
				iCellFoundPerLayer+5,
				primaryVertexContext.mGPUContext.boostCellsFoundForLayer.begin(),
				boostQueues[0]
		);

		iTrackletsNum=primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[0];
		if((iTrackletsNum % workgroupSize)!=0){
			int mult=iTrackletsNum/workgroupSize;
			iTrackletsNum=(mult+1)*workgroupSize;
		}

		int *firstLayerLookUpTable;
		firstLayerLookUpTable=(int*)malloc(iTrackletsNum*sizeof(int));
		memset(firstLayerLookUpTable,0,iTrackletsNum*sizeof(int));
		boost::compute::vector<int> boostFirstLayerCellsLookup(iTrackletsNum,boostContext);
		boost::compute::copy_n(
				firstLayerLookUpTable,
				iTrackletsNum,
				boostFirstLayerCellsLookup.begin(),
				boostQueues[0]
		);


		for (int iLayer{ 0 }; iLayer<Constants::ITS::CellsPerRoad; ++iLayer){
			iTrackletsNum=primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer];
			computeCellsBoostKernel.set_arg(0,primaryVertexContext.mGPUContext.boostPrimaryVertex);
			computeCellsBoostKernel.set_arg(1,primaryVertexContext.mGPUContext.boostTracklets[iLayer]);
			computeCellsBoostKernel.set_arg(2,primaryVertexContext.mGPUContext.boostTracklets[iLayer+1]);
			computeCellsBoostKernel.set_arg(3,primaryVertexContext.mGPUContext.boostClusters[iLayer]);
			computeCellsBoostKernel.set_arg(4,primaryVertexContext.mGPUContext.boostClusters[iLayer+1]);
			computeCellsBoostKernel.set_arg(5,primaryVertexContext.mGPUContext.boostClusters[iLayer+2]);
			computeCellsBoostKernel.set_arg(6,primaryVertexContext.mGPUContext.boostCells[iLayer]);
			computeCellsBoostKernel.set_arg(7,primaryVertexContext.mGPUContext.boostLayerIndex[iLayer]);
			computeCellsBoostKernel.set_arg(8,primaryVertexContext.mGPUContext.boostTrackletsFoundForLayer);
			computeCellsBoostKernel.set_arg(9,primaryVertexContext.mGPUContext.boostTrackletsLookupTable[iLayer]);
			if(iLayer==0)
				computeCellsBoostKernel.set_arg(10, boostFirstLayerCellsLookup);
			else
				computeCellsBoostKernel.set_arg(10, primaryVertexContext.mGPUContext.boostCellsLookupTable[iLayer-1]);
			computeCellsBoostKernel.set_arg(11, primaryVertexContext.mGPUContext.boostCellsFoundForLayer);

			if((iTrackletsNum % workgroupSize)!=0){
				int mult=iTrackletsNum/workgroupSize;
				iTrackletsNum=(mult+1)*workgroupSize;
			}

			boostQueues[iLayer].enqueue_1d_range_kernel(computeCellsBoostKernel,0,iTrackletsNum,workgroupSize);
		}



		//scan
		for (int iLayer { 0 }; iLayer<Constants::ITS::CellsPerRoad; ++iLayer) {
			boostQueues[iLayer].finish();
			iTrackletsNum=primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer];

			if(iLayer==0){

				// sort data on the device
				compute::inclusive_scan(
						boostFirstLayerCellsLookup.begin(),
						boostFirstLayerCellsLookup.end(),
						boostFirstLayerCellsLookup.begin(),
						boostQueues[0]);

				// copy data back to the host
				compute::copy_n(
						boostFirstLayerCellsLookup.begin(),
						iTrackletsNum,
						firstLayerLookUpTable,
						boostQueues[0]);

				primaryVertexContext.mGPUContext.iCellFoundPerLayer[iLayer]=firstLayerLookUpTable[iTrackletsNum-1];
			}
			else{
				// sort data on the device
				compute::inclusive_scan(
						primaryVertexContext.mGPUContext.boostCellsLookupTable[iLayer-1].begin(),
						primaryVertexContext.mGPUContext.boostCellsLookupTable[iLayer-1].begin().operator +=(iTrackletsNum),
						primaryVertexContext.mGPUContext.boostCellsLookupTable[iLayer-1].begin(),
						boostQueues[iLayer]);
				// copy data back to the host
				compute::copy_n(
						primaryVertexContext.mGPUContext.boostCellsLookupTable[iLayer-1].begin(),
						iTrackletsNum,
						primaryVertexContext.mCellsLookupTable[iLayer-1].begin(),
						boostQueues[iLayer]);

				primaryVertexContext.mGPUContext.iCellFoundPerLayer[iLayer]=primaryVertexContext.mCellsLookupTable[iLayer-1][iTrackletsNum-1];
			}
			//std::cout<<"["<<iLayer<<"]: "<<primaryVertexContext.mGPUContext.iCellFoundPerLayer[iLayer]<<std::endl;
		}


		BOOST_COMPUTE_FUNCTION(bool, sort_by_y, (Cell a, Cell b),
		{
			return a.mFirstTrackletIndex < b.mFirstTrackletIndex;
		});

		for (int iLayer{ 0 }; iLayer<Constants::ITS::CellsPerRoad; ++iLayer) {
			int iCellNumber=primaryVertexContext.mGPUContext.iCellFoundPerLayer[iLayer];
			compute::stable_sort(primaryVertexContext.mGPUContext.boostCells[iLayer].begin(),
					primaryVertexContext.mGPUContext.boostCells[iLayer].begin().operator +=(iCellNumber),
					sort_by_y,
					boostQueues[iLayer]);
		}

		//std::cout<<"Cell found"<<std::endl;
		for (int iLayer{ 0 }; iLayer<Constants::ITS::CellsPerRoad; ++iLayer){
			int iCellNumber=primaryVertexContext.mGPUContext.iCellFoundPerLayer[iLayer];
			//std::cout<<"["<<iLayer<<"]: "<<iCellNumber<<std::endl;
			boostQueues[iLayer].finish();
			primaryVertexContext.mCells[iLayer].resize(iCellNumber);
			compute::copy_n(
					primaryVertexContext.mGPUContext.boostCells[iLayer].begin(),
					iCellNumber,
					primaryVertexContext.mCells[iLayer].begin(),
					boostQueues[iLayer]);

			if(iLayer>0){
				int iTrackletsNum=primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer];
				std::copy_n(
						primaryVertexContext.mCellsLookupTable[iLayer-1].begin(),
						iTrackletsNum-1,
						primaryVertexContext.mCellsLookupTable[iLayer-1].begin().operator ++());
				primaryVertexContext.mCellsLookupTable[iLayer-1][0]=0;
			}
		}


		free(firstLayerLookUpTable);

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
