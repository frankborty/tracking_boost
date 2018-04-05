//============================================================================
// Name        : KernelTest.cpp
// Author      : frank
// Version     :
// Copyright   : Your copyright notice
// Description : CountLayerCells opencl kernel
//============================================================================

#include "Definitions.h"
#include "Constants.h"
#include "Tracklet.h"
#include "Cluster.h"
#include "Cell.h"


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
		__global Tracklet* currentLayerTracklets,	//3
		__global Tracklet* nextLayerTracklets,	//4
		__global Cluster* currentLayerClusters,	//5
		__global Cluster* nextLayerClusters,		//6
		__global Cluster* next2LayerClusters,		//7
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
	
	
	
	Tracklet currentTracklet=currentLayerTracklets[currentTrackletIndex];
	int nextLayerClusterIndex=currentTracklet.secondClusterIndex;

	int nextLayerFirstTrackletIndex;
	if(nextLayerClusterIndex==0)
		nextLayerFirstTrackletIndex=0;
	else
		nextLayerFirstTrackletIndex=currentLayerTrackletsLookupTable[nextLayerClusterIndex-1];

	int nextLayerTrackletsNum=iLayerTrackletSize[iLayer + 1];

	Tracklet nextLayerFirstTracklet=nextLayerTracklets[nextLayerFirstTrackletIndex];
	if (nextLayerFirstTracklet.firstClusterIndex == nextLayerClusterIndex) {
		Cluster firstCellCluster=currentLayerClusters[currentTracklet.firstClusterIndex] ;

		Cluster secondCellCluster=nextLayerClusters[currentTracklet.secondClusterIndex];

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
			
			Tracklet nextTracklet=nextLayerTracklets[iNextLayerTracklet];

			float deltaTanLambda=fabs(currentTracklet.tanLambda - nextTracklet.tanLambda);

			float deltaPhi=fabs(currentTracklet.phiCoordinate - nextTracklet.phiCoordinate);

			if (deltaTanLambda < CellMaxDeltaTanLambdaThreshold && (deltaPhi < CellMaxDeltaPhiThreshold
				|| fabs(deltaPhi - TwoPi) < CellMaxDeltaPhiThreshold)) {

				float averageTanLambda= 0.5f * (currentTracklet.tanLambda + nextTracklet.tanLambda) ;

				float directionZIntersection=-averageTanLambda * firstCellCluster.rCoordinate+ firstCellCluster.zCoordinate ;

				float deltaZ=fabs(directionZIntersection - primaryVertex.z) ;

				if (deltaZ < CellMaxDeltaZThreshold[iLayer]) {

					Cluster thirdCellCluster=next2LayerClusters[nextTracklet.secondClusterIndex];

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

