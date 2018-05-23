//============================================================================
// Name        : KernelTest.cpp
// Author      : frank
// Version     :
// Copyright   : Your copyright notice
// Description : ComputeLayerCells opencl kernel
//============================================================================

#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/Constants.h"
#include "ITSReconstruction/CA/Tracklet.h"
#include "ITSReconstruction/CA/Cluster.h"
#include "ITSReconstruction/CA/Cell.h"


Float3Struct crossProduct(Float3Struct* firstVector,  Float3Struct* secondVector)
{
	Float3Struct result;
	result.x=(firstVector->y * secondVector->z) - (firstVector->z * secondVector->y);
	result.y=(firstVector->z * secondVector->x) - (firstVector->x * secondVector->z);
    result.z=(firstVector->x * secondVector->y) - (firstVector->y * secondVector->x);
    return result;
}


__kernel void computeLayerCells(
		__global Float3Struct* fPrimaryVertex,	//0
		__global int *iCurrentLayer,	//1
		__global int * iLayerTrackletSize, //2 store the number of tracklet found for each layer
		__global Tracklet* currentLayerTracklets,	//3
		__global Tracklet* nextLayerTracklets,	//4
		__global Cluster* currentLayerClusters,	//5
		__global Cluster* nextLayerClusters,		//6
		__global Cluster* next2LayerClusters,		//7
		__global int* currentLayerTrackletsLookupTable, //8
		__global int * iCellsPerTrackletPreviousLayer, 	//9
		__global Cell* currentLayerCells			//10
)
{
	const int currentTrackletIndex=get_global_id(0);
	const Float3Struct primaryVertex = *fPrimaryVertex;
	int iLayer=*iCurrentLayer;
	if (currentTrackletIndex >= iLayerTrackletSize[iLayer])
		return; 
	int itmp=0;
	int trackletCellsNum = 0;
	
	int currentLookUpValue=iCellsPerTrackletPreviousLayer[currentTrackletIndex];
	int previousLookUpValue=0;
	if(currentTrackletIndex!=0)
		previousLookUpValue=iCellsPerTrackletPreviousLayer[currentTrackletIndex-1];
	
	int numberOfCellsToFind=currentLookUpValue-previousLookUpValue;
	if(numberOfCellsToFind==0)
		return;
	
	
	
	Tracklet currentTracklet=currentLayerTracklets[currentTrackletIndex];
	const int nextLayerClusterIndex=currentTracklet.secondClusterIndex;

	int nextLayerFirstTrackletIndex;
	if(nextLayerClusterIndex==0)
		nextLayerFirstTrackletIndex=0;
	else
		nextLayerFirstTrackletIndex=currentLayerTrackletsLookupTable[nextLayerClusterIndex-1];

	const int nextLayerTrackletsNum=iLayerTrackletSize[iLayer + 1];

	Tracklet nextLayerFirstTracklet=nextLayerTracklets[nextLayerFirstTrackletIndex];
	if (nextLayerFirstTracklet.firstClusterIndex == nextLayerClusterIndex) {
		Cluster firstCellCluster=currentLayerClusters[currentTracklet.firstClusterIndex] ;

		Cluster secondCellCluster=nextLayerClusters[currentTracklet.secondClusterIndex];

		const float firstCellClusterQuadraticRCoordinate=firstCellCluster.rCoordinate * firstCellCluster.rCoordinate;

		const float secondCellClusterQuadraticRCoordinate=secondCellCluster.rCoordinate * secondCellCluster.rCoordinate;


		Float3Struct firstDeltaVector;
		firstDeltaVector.x=secondCellCluster.xCoordinate - firstCellCluster.xCoordinate;
		firstDeltaVector.y=secondCellCluster.yCoordinate - firstCellCluster.yCoordinate;
		firstDeltaVector.z=secondCellClusterQuadraticRCoordinate- firstCellClusterQuadraticRCoordinate;
		
		for (int iNextLayerTracklet=nextLayerFirstTrackletIndex ;
				iNextLayerTracklet < nextLayerTrackletsNum	
				&& nextLayerTracklets[iNextLayerTracklet].firstClusterIndex== nextLayerClusterIndex;
				++iNextLayerTracklet){
			
			Tracklet nextTracklet=nextLayerTracklets[iNextLayerTracklet];

			const float deltaTanLambda=fabs(currentTracklet.tanLambda - nextTracklet.tanLambda);

			const float deltaPhi=fabs(currentTracklet.phiCoordinate - nextTracklet.phiCoordinate);

			if (deltaTanLambda < CellMaxDeltaTanLambdaThreshold && (deltaPhi < CellMaxDeltaPhiThreshold
				|| fabs(deltaPhi - TwoPi) < CellMaxDeltaPhiThreshold)) {

				const float averageTanLambda= 0.5f * (currentTracklet.tanLambda + nextTracklet.tanLambda) ;

				const float directionZIntersection=-averageTanLambda * firstCellCluster.rCoordinate+ firstCellCluster.zCoordinate ;

				const float deltaZ=fabs(directionZIntersection - primaryVertex.z) ;

				if (deltaZ < CellMaxDeltaZThreshold[iLayer]) {

					Cluster thirdCellCluster=next2LayerClusters[nextTracklet.secondClusterIndex];

					const float thirdCellClusterQuadraticRCoordinate=thirdCellCluster.rCoordinate	* thirdCellCluster.rCoordinate ;

					Float3Struct secondDeltaVector;
					secondDeltaVector.x=thirdCellCluster.xCoordinate - firstCellCluster.xCoordinate;
					secondDeltaVector.y=thirdCellCluster.yCoordinate - firstCellCluster.yCoordinate;
					secondDeltaVector.z=thirdCellClusterQuadraticRCoordinate- firstCellClusterQuadraticRCoordinate;

					Float3Struct cellPlaneNormalVector=crossProduct(&firstDeltaVector, &secondDeltaVector);

					const float vectorNorm=sqrt(cellPlaneNormalVector.x * cellPlaneNormalVector.x + cellPlaneNormalVector.y * cellPlaneNormalVector.y
							+ cellPlaneNormalVector.z * cellPlaneNormalVector.z);



					if (!(vectorNorm < FloatMinThreshold || fabs(cellPlaneNormalVector.z) < FloatMinThreshold)) {
						const float inverseVectorNorm = 1.0f / vectorNorm ;

						const Float3Struct normalizedPlaneVector = {cellPlaneNormalVector.x * inverseVectorNorm, cellPlaneNormalVector.y
							*inverseVectorNorm, cellPlaneNormalVector.z * inverseVectorNorm };

						const float planeDistance = -normalizedPlaneVector.x * (secondCellCluster.xCoordinate - primaryVertex.x)
							- (normalizedPlaneVector.y * secondCellCluster.yCoordinate - primaryVertex.y)
							- normalizedPlaneVector.z * secondCellClusterQuadraticRCoordinate ;



						const float normalizedPlaneVectorQuadraticZCoordinate = normalizedPlaneVector.z * normalizedPlaneVector.z ;


						
						const float cellTrajectoryRadius = sqrt(
							(1.0f - normalizedPlaneVectorQuadraticZCoordinate - 4.0f * planeDistance * normalizedPlaneVector.z)
							/ (4.0f * normalizedPlaneVectorQuadraticZCoordinate)) ;



						const Float2Struct circleCenter ={ -0.5f * normalizedPlaneVector.x / normalizedPlaneVector.z, -0.5f
							* normalizedPlaneVector.y / normalizedPlaneVector.z };


						const float distanceOfClosestApproach = fabs(
							cellTrajectoryRadius - sqrt(circleCenter.x * circleCenter.x + circleCenter.y * circleCenter.y)) ;


						if (distanceOfClosestApproach	<= CellMaxDistanceOfClosestApproachThreshold[iLayer]) {
							__global Cell* cell=&currentLayerCells[previousLookUpValue];
							cell->mFirstClusterIndex=currentTracklet.firstClusterIndex;
							cell->mSecondClusterIndex=currentTracklet.secondClusterIndex;
							cell->mThirdClusterIndex=nextTracklet.secondClusterIndex;
							cell->mFirstTrackletIndex=currentTrackletIndex;
							cell->mSecondTrackletIndex=iNextLayerTracklet;
							cell->mNormalVectorCoordinates.x=normalizedPlaneVector.x;
							cell->mNormalVectorCoordinates.y=normalizedPlaneVector.y;
							cell->mNormalVectorCoordinates.z=normalizedPlaneVector.z;
							cell->mCurvature=1.0f/cellTrajectoryRadius;
							cell->mLevel=1;
							
							previousLookUpValue++;
							if(currentLookUpValue==previousLookUpValue)
					  			return;	
									
							
						}
					}
				}
			}
		}
			

	}
	
}