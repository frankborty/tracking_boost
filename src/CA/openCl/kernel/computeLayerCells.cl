__constant float CellMaxDeltaZThreshold[5]= { 0.2f, 0.4f, 0.5f, 0.6f, 3.0f } ;
__constant float TrackletMaxDeltaZThreshold[6]= { 0.1f, 0.1f, 0.3f, 0.3f, 0.3f, 0.3f }; //default
//__constant float TrackletMaxDeltaZThreshold[6]= { 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f };
__constant int ZBins=20;
__constant int PhiBins=20;
__constant float Pi=3.14159265359f;
__constant float TwoPi=2.0f * 3.14159265359f ;
__constant int UnusedIndex=-1 ;
__constant float CellMaxDeltaPhiThreshold=0.14f;
__constant float CellMaxDeltaTanLambdaThreshold=0.025f;
__constant float PhiCoordinateCut=0.3f;	//default
//__constant float PhiCoordinateCut=PHICUT;	//default

__constant float FloatMinThreshold = 1e-20f ;
__constant float ZCoordinateCut=0.5f;	//default
__constant float InversePhiBinSize=20 / (2.0f * 3.14159265359f) ;
__constant float LayersZCoordinate[7]={16.333f, 16.333f, 16.333f, 42.140f, 42.140f, 73.745f, 73.745f};
__constant float LayersRCoordinate[7]={2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f, 34.388f, 39.3329f};
__constant float InverseZBinSize[7]=  {0.5f * 20 / 16.333f, 0.5f * 20 / 16.333f, 0.5f * 20 / 16.333f,0.5f * 20 / 42.140f, 0.5f * 20 / 42.140f, 0.5f * 20 / 73.745f, 0.5f * 20 / 73.745f };
__constant float CellMaxDistanceOfClosestApproachThreshold[5]= { 0.05f, 0.04f, 0.05f, 0.2f, 0.4f } ;


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


int myMin(int a,int b){
	if(a<b)
		return a;
	return b;
}

int myMax(int a,int b){
	if(a>b)
		return a;
	return b;
}

double myAbs(double iNum){
	if(iNum<0)
		return -1*iNum;
	return iNum;
}



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


Int4Struct getBinsRect(__global ClusterStruct* currentCluster, int layerIndex,float directionZIntersection)
{
	const float zRangeMin = directionZIntersection - 2 * ZCoordinateCut;
	const float phiRangeMin = currentCluster->phiCoordinate - PhiCoordinateCut;
	const float zRangeMax = directionZIntersection + 2 * ZCoordinateCut;
	const float phiRangeMax = currentCluster->phiCoordinate + PhiCoordinateCut;
	Int4Struct binRect;
	binRect.x=0;
	binRect.y=0;
	binRect.z=0;
	binRect.w=0;

	if (zRangeMax < -LayersZCoordinate[layerIndex + 1]|| zRangeMin > LayersZCoordinate[layerIndex + 1] || zRangeMin > zRangeMax) {
		return binRect;
	}

	binRect.x=myMax(0, getZBinIndex(layerIndex + 1, zRangeMin));
	binRect.y=getPhiBinIndex(getNormalizedPhiCoordinate(phiRangeMin));
	binRect.z=myMin(ZBins - 1, getZBinIndex(layerIndex + 1, zRangeMax));
	binRect.w=getPhiBinIndex(getNormalizedPhiCoordinate(phiRangeMax));
	return binRect;

}


int getBinIndex(int zIndex,int phiIndex)
{
	return myMin(phiIndex * PhiBins + zIndex,ZBins * PhiBins);
}

Float3Struct crossProduct(const Float3Struct* firstVector,  const Float3Struct* secondVector)
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
		__global int * iCellsPerTrackletPreviousLayer, 	//9
		__global int * iCurrentCellCounter //10 store the total number of cell found 
)
{	
	const int currentTrackletIndex=get_global_id(0);
	const Float3Struct primaryVertex = *fPrimaryVertex;
	int iLayer=*iCurrentLayer;
	int itmp=0;
	int trackletCellsNum = 0;
	iCellsPerTrackletPreviousLayer[currentTrackletIndex]=0;
	
	if (currentTrackletIndex >= iLayerTrackletSize[iLayer])
		return; 
	
	global TrackletStruct* currentTracklet=&currentLayerTracklets[currentTrackletIndex];
	const int nextLayerClusterIndex=currentTracklet->secondClusterIndex;

	const int nextLayerFirstTrackletIndex=currentLayerTrackletsLookupTable[nextLayerClusterIndex];

	const int nextLayerTrackletsNum=iLayerTrackletSize[iLayer + 1];

	global TrackletStruct* nextLayerFirstTracklet=&nextLayerTracklets[nextLayerFirstTrackletIndex];
	if (nextLayerFirstTracklet->firstClusterIndex == nextLayerClusterIndex) {
		__global ClusterStruct* firstCellCluster=&currentLayerClusters[currentTracklet->firstClusterIndex] ;

		__global ClusterStruct* secondCellCluster=&nextLayerClusters[currentTracklet->secondClusterIndex];

		const float firstCellClusterQuadraticRCoordinate=firstCellCluster->rCoordinate * firstCellCluster->rCoordinate;

		const float secondCellClusterQuadraticRCoordinate=secondCellCluster->rCoordinate * secondCellCluster->rCoordinate;


		Float3Struct firstDeltaVector;
		firstDeltaVector.x=secondCellCluster->xCoordinate - firstCellCluster->xCoordinate;
		firstDeltaVector.y=secondCellCluster->yCoordinate - firstCellCluster->yCoordinate;
		firstDeltaVector.z=secondCellClusterQuadraticRCoordinate- firstCellClusterQuadraticRCoordinate;
		
		for (int iNextLayerTracklet=nextLayerFirstTrackletIndex ;
				iNextLayerTracklet < nextLayerTrackletsNum	
				&& nextLayerTracklets[iNextLayerTracklet].firstClusterIndex== nextLayerClusterIndex;
				++iNextLayerTracklet){
			
			__global TrackletStruct* nextTracklet=&nextLayerTracklets[iNextLayerTracklet];

			const float deltaTanLambda=myAbs(currentTracklet->tanLambda - nextTracklet->tanLambda);

			const float deltaPhi=myAbs(currentTracklet->phiCoordinate - nextTracklet->phiCoordinate);

			if (deltaTanLambda < CellMaxDeltaTanLambdaThreshold && (deltaPhi < CellMaxDeltaPhiThreshold
				|| myAbs(deltaPhi - TwoPi) < CellMaxDeltaPhiThreshold)) {

				const float averageTanLambda= 0.5f * (currentTracklet->tanLambda + nextTracklet->tanLambda) ;

				const float directionZIntersection=-averageTanLambda * firstCellCluster->rCoordinate+ firstCellCluster->zCoordinate ;

				const float deltaZ=myAbs(directionZIntersection - primaryVertex.z) ;

				if (deltaZ < CellMaxDeltaZThreshold[iLayer]) {

					__global ClusterStruct* thirdCellCluster=&next2LayerClusters[nextTracklet->secondClusterIndex];

					const float thirdCellClusterQuadraticRCoordinate=thirdCellCluster->rCoordinate	* thirdCellCluster->rCoordinate ;

					Float3Struct secondDeltaVector;
					secondDeltaVector.x=thirdCellCluster->xCoordinate - firstCellCluster->xCoordinate;
					secondDeltaVector.y=thirdCellCluster->yCoordinate - firstCellCluster->yCoordinate;
					secondDeltaVector.z=thirdCellClusterQuadraticRCoordinate- firstCellClusterQuadraticRCoordinate;

					Float3Struct cellPlaneNormalVector=crossProduct(&firstDeltaVector, &secondDeltaVector);

					const float vectorNorm=sqrt(cellPlaneNormalVector.x * cellPlaneNormalVector.x + cellPlaneNormalVector.y * cellPlaneNormalVector.y
							+ cellPlaneNormalVector.z * cellPlaneNormalVector.z);



					if (!(vectorNorm < FloatMinThreshold || myAbs(cellPlaneNormalVector.z) < FloatMinThreshold)) {
						const float inverseVectorNorm = 1.0f / vectorNorm ;

						const Float3Struct normalizedPlaneVector = {cellPlaneNormalVector.x * inverseVectorNorm, cellPlaneNormalVector.y
							*inverseVectorNorm, cellPlaneNormalVector.z * inverseVectorNorm };

						const float planeDistance = -normalizedPlaneVector.x * (secondCellCluster->xCoordinate - primaryVertex.x)
							- (normalizedPlaneVector.y * secondCellCluster->yCoordinate - primaryVertex.y)
							- normalizedPlaneVector.z * secondCellClusterQuadraticRCoordinate ;



						const float normalizedPlaneVectorQuadraticZCoordinate = normalizedPlaneVector.z * normalizedPlaneVector.z ;


						
						const float cellTrajectoryRadius = sqrt(
							(1.0f - normalizedPlaneVectorQuadraticZCoordinate - 4.0f * planeDistance * normalizedPlaneVector.z)
							/ (4.0f * normalizedPlaneVectorQuadraticZCoordinate)) ;



						const Float2Struct circleCenter ={ -0.5f * normalizedPlaneVector.x / normalizedPlaneVector.z, -0.5f
							* normalizedPlaneVector.y / normalizedPlaneVector.z };


						const float distanceOfClosestApproach = myAbs(
							cellTrajectoryRadius - sqrt(circleCenter.x * circleCenter.x + circleCenter.y * circleCenter.y)) ;


						if (distanceOfClosestApproach	<= CellMaxDistanceOfClosestApproachThreshold[iLayer]) {
							atom_inc(&iCurrentCellCounter[iLayer]);
							++trackletCellsNum;		
							
						}
					}
				}
			}
		}
			
		if(trackletCellsNum>0) {
			iCellsPerTrackletPreviousLayer[currentTrackletIndex] = trackletCellsNum;
		}

	}
}



__kernel void computeLayerCells(
		__global Float3Struct* fPrimaryVertex,	//0
		__global int *iCurrentLayer,	//1
		__global int * iLayerTrackletSize, //2 store the number of tracklet found for each layer
		__global TrackletStruct* currentLayerTracklets,	//3
		__global TrackletStruct* nextLayerTracklets,	//4
		__global ClusterStruct* currentLayerClusters,	//5
		__global ClusterStruct* nextLayerClusters,		//6
		__global ClusterStruct* next2LayerClusters,		//7
		__global int* currentLayerTrackletsLookupTable, //8
		__global int * iCellsPerTrackletPreviousLayer, 	//9
		__global CellStruct* currentLayerCells			//10
)
{
	const int currentTrackletIndex=get_global_id(0);
	const Float3Struct primaryVertex = *fPrimaryVertex;
	int iLayer=*iCurrentLayer;
	int itmp=0;
	int trackletCellsNum = 0;
	iCellsPerTrackletPreviousLayer[currentTrackletIndex]=0;
	int currentLookUpValue=iCellsPerTrackletPreviousLayer[currentTrackletIndex];
	int nextLookUpValue=iCellsPerTrackletPreviousLayer[currentTrackletIndex+1];
	int numberOfCellsToFind=nextLookUpValue-currentLookUpValue;
	if(currentLookUpValue==nextLookUpValue)
		return;
	
	if (currentTrackletIndex >= iLayerTrackletSize[iLayer])
		return; 
	
	global TrackletStruct* currentTracklet=&currentLayerTracklets[currentTrackletIndex];
	const int nextLayerClusterIndex=currentTracklet->secondClusterIndex;

	const int nextLayerFirstTrackletIndex=currentLayerTrackletsLookupTable[nextLayerClusterIndex];

	const int nextLayerTrackletsNum=iLayerTrackletSize[iLayer + 1];

	global TrackletStruct* nextLayerFirstTracklet=&nextLayerTracklets[nextLayerFirstTrackletIndex];
	if (nextLayerFirstTracklet->firstClusterIndex == nextLayerClusterIndex) {
		__global ClusterStruct* firstCellCluster=&currentLayerClusters[currentTracklet->firstClusterIndex] ;

		__global ClusterStruct* secondCellCluster=&nextLayerClusters[currentTracklet->secondClusterIndex];

		const float firstCellClusterQuadraticRCoordinate=firstCellCluster->rCoordinate * firstCellCluster->rCoordinate;

		const float secondCellClusterQuadraticRCoordinate=secondCellCluster->rCoordinate * secondCellCluster->rCoordinate;


		Float3Struct firstDeltaVector;
		firstDeltaVector.x=secondCellCluster->xCoordinate - firstCellCluster->xCoordinate;
		firstDeltaVector.y=secondCellCluster->yCoordinate - firstCellCluster->yCoordinate;
		firstDeltaVector.z=secondCellClusterQuadraticRCoordinate- firstCellClusterQuadraticRCoordinate;
		
		for (int iNextLayerTracklet=nextLayerFirstTrackletIndex ;
				iNextLayerTracklet < nextLayerTrackletsNum	
				&& nextLayerTracklets[iNextLayerTracklet].firstClusterIndex== nextLayerClusterIndex;
				++iNextLayerTracklet){
			
			__global TrackletStruct* nextTracklet=&nextLayerTracklets[iNextLayerTracklet];

			const float deltaTanLambda=myAbs(currentTracklet->tanLambda - nextTracklet->tanLambda);

			const float deltaPhi=myAbs(currentTracklet->phiCoordinate - nextTracklet->phiCoordinate);

			if (deltaTanLambda < CellMaxDeltaTanLambdaThreshold && (deltaPhi < CellMaxDeltaPhiThreshold
				|| myAbs(deltaPhi - TwoPi) < CellMaxDeltaPhiThreshold)) {

				const float averageTanLambda= 0.5f * (currentTracklet->tanLambda + nextTracklet->tanLambda) ;

				const float directionZIntersection=-averageTanLambda * firstCellCluster->rCoordinate+ firstCellCluster->zCoordinate ;

				const float deltaZ=myAbs(directionZIntersection - primaryVertex.z) ;

				if (deltaZ < CellMaxDeltaZThreshold[iLayer]) {

					__global ClusterStruct* thirdCellCluster=&next2LayerClusters[nextTracklet->secondClusterIndex];

					const float thirdCellClusterQuadraticRCoordinate=thirdCellCluster->rCoordinate	* thirdCellCluster->rCoordinate ;

					Float3Struct secondDeltaVector;
					secondDeltaVector.x=thirdCellCluster->xCoordinate - firstCellCluster->xCoordinate;
					secondDeltaVector.y=thirdCellCluster->yCoordinate - firstCellCluster->yCoordinate;
					secondDeltaVector.z=thirdCellClusterQuadraticRCoordinate- firstCellClusterQuadraticRCoordinate;

					Float3Struct cellPlaneNormalVector=crossProduct(&firstDeltaVector, &secondDeltaVector);

					const float vectorNorm=sqrt(cellPlaneNormalVector.x * cellPlaneNormalVector.x + cellPlaneNormalVector.y * cellPlaneNormalVector.y
							+ cellPlaneNormalVector.z * cellPlaneNormalVector.z);



					if (!(vectorNorm < FloatMinThreshold || myAbs(cellPlaneNormalVector.z) < FloatMinThreshold)) {
						const float inverseVectorNorm = 1.0f / vectorNorm ;

						const Float3Struct normalizedPlaneVector = {cellPlaneNormalVector.x * inverseVectorNorm, cellPlaneNormalVector.y
							*inverseVectorNorm, cellPlaneNormalVector.z * inverseVectorNorm };

						const float planeDistance = -normalizedPlaneVector.x * (secondCellCluster->xCoordinate - primaryVertex.x)
							- (normalizedPlaneVector.y * secondCellCluster->yCoordinate - primaryVertex.y)
							- normalizedPlaneVector.z * secondCellClusterQuadraticRCoordinate ;



						const float normalizedPlaneVectorQuadraticZCoordinate = normalizedPlaneVector.z * normalizedPlaneVector.z ;


						
						const float cellTrajectoryRadius = sqrt(
							(1.0f - normalizedPlaneVectorQuadraticZCoordinate - 4.0f * planeDistance * normalizedPlaneVector.z)
							/ (4.0f * normalizedPlaneVectorQuadraticZCoordinate)) ;



						const Float2Struct circleCenter ={ -0.5f * normalizedPlaneVector.x / normalizedPlaneVector.z, -0.5f
							* normalizedPlaneVector.y / normalizedPlaneVector.z };


						const float distanceOfClosestApproach = myAbs(
							cellTrajectoryRadius - sqrt(circleCenter.x * circleCenter.x + circleCenter.y * circleCenter.y)) ;


						if (distanceOfClosestApproach	<= CellMaxDistanceOfClosestApproachThreshold[iLayer]) {
							__global CellStruct* cell=&currentLayerCells[currentLookUpValue];
							cell->mFirstClusterIndex=currentTracklet->firstClusterIndex;
							cell->mSecondClusterIndex=currentTracklet->secondClusterIndex;
							cell->mThirdClusterIndex=nextTracklet->secondClusterIndex;
							cell->mFirstTrackletIndex=currentTrackletIndex;
							cell->mSecondTrackletIndex=iNextLayerTracklet;
							cell->mNormalVectorCoordinates.x=normalizedPlaneVector.x;
							cell->mNormalVectorCoordinates.y=normalizedPlaneVector.y;
							cell->mNormalVectorCoordinates.z=normalizedPlaneVector.z;
							cell->mCurvature=1.0f/cellTrajectoryRadius;
							cell->mLevel=1;
							
							currentLookUpValue++;
							if(currentLookUpValue==nextLookUpValue)
					  			return;	
									
							
						}
					}
				}
			}
		}
			

	}
	
}
