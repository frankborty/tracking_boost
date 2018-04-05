//============================================================================
// Name        : KernelTest.cpp
// Author      : frank
// Version     :
// Copyright   : Your copyright notice
// Description : ComputeLayerCells opencl kernel
//============================================================================

#include "Definitions.h"
#include "Constants.h"



	typedef struct{
		int x;
		int y;
		int z;
		int w;
	}Int4Struct;

	typedef struct{
		float x;
		float y;
		float z;
	}Float3Struct;

	typedef struct{
			float x;
			float y;
	}Float2Struct;

	typedef struct{
		float xCoordinate;
		float yCoordinate;
		float zCoordinate;
		float phiCoordinate;
		float rCoordinate;
		int clusterId;
		float alphaAngle;
		int monteCarloId;
		int indexTableBinIndex;
	}ClusterStruct;

	typedef struct{
		int mFirstClusterIndex;
		int mSecondClusterIndex;
		int mThirdClusterIndex;
		int mFirstTrackletIndex;
		int mSecondTrackletIndex;
		Float3Struct mNormalVectorCoordinates;
		float mCurvature;
		int mLevel;
	}CellStruct;

	typedef struct{
		int firstClusterIndex;
		int secondClusterIndex;
		float tanLambda;
		float phiCoordinate;
	}TrackletStruct;

	typedef struct{
		int firstClusterIndex;
		int secondClusterIndex;
		float tanLambda;
		float phiCoordinate;
	}RoadsStruct;

	typedef struct{
		void * srPunt;
		int size;
	}VectStruct;



int getZBinIndex(int layerIndex, float zCoordinate){
	return (zCoordinate + LayersZCoordinate[layerIndex])* InverseZBinSize[layerIndex];
}

int getPhiBinIndex(float currentPhi)
{
  return (currentPhi * InversePhiBinSize);
}

float getNormalizedPhiCoordinate(float phiCoordinate)
{
  return (phiCoordinate < 0) ? phiCoordinate + TwoPi :
         (phiCoordinate > TwoPi) ? phiCoordinate - TwoPi : phiCoordinate;
}


Int4Struct getBinsRect(ClusterStruct currentCluster, int layerIndex,float directionZIntersection)
{
	const float zRangeMin = directionZIntersection - 2 * ZCoordinateCut;
	const float phiRangeMin = currentCluster.phiCoordinate - PhiCoordinateCut;
	const float zRangeMax = directionZIntersection + 2 * ZCoordinateCut;
	const float phiRangeMax = currentCluster.phiCoordinate + PhiCoordinateCut;
	Int4Struct binRect;
	binRect.x=0;
	binRect.y=0;
	binRect.z=0;
	binRect.w=0;

	if (zRangeMax < -LayersZCoordinate[layerIndex + 1]|| zRangeMin > LayersZCoordinate[layerIndex + 1] || zRangeMin > zRangeMax) {
		return binRect;
	}

	binRect.x=max(0, getZBinIndex(layerIndex + 1, zRangeMin));
	binRect.y=getPhiBinIndex(getNormalizedPhiCoordinate(phiRangeMin));
	binRect.z=min(ZBins - 1, getZBinIndex(layerIndex + 1, zRangeMax));
	binRect.w=getPhiBinIndex(getNormalizedPhiCoordinate(phiRangeMax));
	return binRect;

}

int getBinIndex(int zIndex,int phiIndex)
{
	return min(phiIndex * PhiBins + zIndex,ZBins * PhiBins);
}

Float3Struct crossProduct(Float3Struct* firstVector,  Float3Struct* secondVector)
{
	Float3Struct result;
	result.x=(firstVector->y * secondVector->z) - (firstVector->z * secondVector->y);
	result.y=(firstVector->z * secondVector->x) - (firstVector->x * secondVector->z);
    result.z=(firstVector->x * secondVector->y) - (firstVector->y * secondVector->x);
    return result;
}



__kernel void countLayerCells(
		__global Float3Struct* fPrimaryVertex,	//0
		__global int *iCurrentLayer,	//1
		__global int * iLayerTrackletSize, //2 store the number of tracklet found for each layer
		__global TrackletStruct* currentLayerTracklets,	//3
		__global TrackletStruct* nextLayerTracklets,	//4
		__global ClusterStruct* currentLayerClusters,	//5
		__global ClusterStruct* nextLayerClusters,		//6
		__global ClusterStruct* next2LayerClusters,		//7
		__global int* currentLayerTrackletsLookupTable, //8
		__global int * iCellsPerTrackletPreviousLayer 	//9
)
{	
	const int currentTrackletIndex=get_global_id(0);
	const Float3Struct primaryVertex = *fPrimaryVertex;
	int iLayer=*iCurrentLayer;
	if (currentTrackletIndex >= iLayerTrackletSize[iLayer])
		return; 
	int itmp=0;
	int trackletCellsNum = 0;
	iCellsPerTrackletPreviousLayer[currentTrackletIndex]=0;
	
	
	
	TrackletStruct currentTracklet=currentLayerTracklets[currentTrackletIndex];
	int nextLayerClusterIndex=currentTracklet.secondClusterIndex;

	int nextLayerFirstTrackletIndex;
	if(nextLayerClusterIndex==0)
		nextLayerFirstTrackletIndex=0;
	else
		nextLayerFirstTrackletIndex=currentLayerTrackletsLookupTable[nextLayerClusterIndex-1];

	int nextLayerTrackletsNum=iLayerTrackletSize[iLayer + 1];

	TrackletStruct nextLayerFirstTracklet=nextLayerTracklets[nextLayerFirstTrackletIndex];
	if (nextLayerFirstTracklet.firstClusterIndex == nextLayerClusterIndex) {
		ClusterStruct firstCellCluster=currentLayerClusters[currentTracklet.firstClusterIndex] ;

		ClusterStruct secondCellCluster=nextLayerClusters[currentTracklet.secondClusterIndex];

		float firstCellClusterQuadraticRCoordinate=firstCellCluster.rCoordinate * firstCellCluster.rCoordinate;

		float secondCellClusterQuadraticRCoordinate=secondCellCluster.rCoordinate * secondCellCluster.rCoordinate;


		Float3Struct firstDeltaVector;
		firstDeltaVector.x=secondCellCluster.xCoordinate - firstCellCluster.xCoordinate;
		firstDeltaVector.y=secondCellCluster.yCoordinate - firstCellCluster.yCoordinate;
		firstDeltaVector.z=secondCellClusterQuadraticRCoordinate- firstCellClusterQuadraticRCoordinate;
		
		for (int iNextLayerTracklet=nextLayerFirstTrackletIndex ;
				iNextLayerTracklet < nextLayerTrackletsNum	
				&& nextLayerTracklets[iNextLayerTracklet].firstClusterIndex== nextLayerClusterIndex;
				++iNextLayerTracklet){
			
			TrackletStruct nextTracklet=nextLayerTracklets[iNextLayerTracklet];

			float deltaTanLambda=fabs(currentTracklet.tanLambda - nextTracklet.tanLambda);

			float deltaPhi=fabs(currentTracklet.phiCoordinate - nextTracklet.phiCoordinate);

			if (deltaTanLambda < CellMaxDeltaTanLambdaThreshold && (deltaPhi < CellMaxDeltaPhiThreshold
				|| fabs(deltaPhi - TwoPi) < CellMaxDeltaPhiThreshold)) {

				float averageTanLambda= 0.5f * (currentTracklet.tanLambda + nextTracklet.tanLambda) ;

				float directionZIntersection=-averageTanLambda * firstCellCluster.rCoordinate+ firstCellCluster.zCoordinate ;

				float deltaZ=fabs(directionZIntersection - primaryVertex.z) ;

				if (deltaZ < CellMaxDeltaZThreshold[iLayer]) {

					ClusterStruct thirdCellCluster=next2LayerClusters[nextTracklet.secondClusterIndex];

					float thirdCellClusterQuadraticRCoordinate=thirdCellCluster.rCoordinate	* thirdCellCluster.rCoordinate ;

					Float3Struct secondDeltaVector;
					secondDeltaVector.x=thirdCellCluster.xCoordinate - firstCellCluster.xCoordinate;
					secondDeltaVector.y=thirdCellCluster.yCoordinate - firstCellCluster.yCoordinate;
					secondDeltaVector.z=thirdCellClusterQuadraticRCoordinate- firstCellClusterQuadraticRCoordinate;

					Float3Struct cellPlaneNormalVector=crossProduct(&firstDeltaVector, &secondDeltaVector);

					float vectorNorm=sqrt(cellPlaneNormalVector.x * cellPlaneNormalVector.x + cellPlaneNormalVector.y * cellPlaneNormalVector.y
							+ cellPlaneNormalVector.z * cellPlaneNormalVector.z);



					if (!(vectorNorm < FloatMinThreshold || fabs(cellPlaneNormalVector.z) < FloatMinThreshold)) {
						float inverseVectorNorm = 1.0f / vectorNorm ;

						Float3Struct normalizedPlaneVector = {cellPlaneNormalVector.x * inverseVectorNorm, cellPlaneNormalVector.y
							*inverseVectorNorm, cellPlaneNormalVector.z * inverseVectorNorm };

						float planeDistance = -normalizedPlaneVector.x * (secondCellCluster.xCoordinate - primaryVertex.x)
							- (normalizedPlaneVector.y * secondCellCluster.yCoordinate - primaryVertex.y)
							- normalizedPlaneVector.z * secondCellClusterQuadraticRCoordinate ;



						float normalizedPlaneVectorQuadraticZCoordinate = normalizedPlaneVector.z * normalizedPlaneVector.z ;


						
						float cellTrajectoryRadius = sqrt(
							(1.0f - normalizedPlaneVectorQuadraticZCoordinate - 4.0f * planeDistance * normalizedPlaneVector.z)
							/ (4.0f * normalizedPlaneVectorQuadraticZCoordinate)) ;



						Float2Struct circleCenter ={ -0.5f * normalizedPlaneVector.x / normalizedPlaneVector.z, -0.5f
							* normalizedPlaneVector.y / normalizedPlaneVector.z };


						float distanceOfClosestApproach = fabs(
							cellTrajectoryRadius - sqrt(circleCenter.x * circleCenter.x + circleCenter.y * circleCenter.y)) ;


						if (distanceOfClosestApproach	<= CellMaxDistanceOfClosestApproachThreshold[iLayer]) {
							++trackletCellsNum;		
							
						}
					}
				}
			}
		}
			
		if(trackletCellsNum>0) {
			iCellsPerTrackletPreviousLayer[currentTrackletIndex] = trackletCellsNum;
		}
		else {
			iCellsPerTrackletPreviousLayer[currentTrackletIndex] = 0;
		}
	}
}

