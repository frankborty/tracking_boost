//============================================================================
// Name        : KernelTest.cpp
// Author      : frank
// Version     :
// Copyright   : Your copyright notice
// Description : ComputeLayerTracklets opencl kernel
//============================================================================


__constant float TrackletMaxDeltaZThreshold[6]= { 0.1f, 0.1f, 0.3f, 0.3f, 0.3f, 0.3f }; //default
__constant int ZBins=20;
__constant int PhiBins=20;
__constant float Pi=3.14159265359f;
__constant float TwoPi=2.0f * 3.14159265359f ;
__constant int UnusedIndex=-1 ;
__constant float CellMaxDeltaPhiThreshold=0.14f;
__constant float PhiCoordinateCut=0.3f;	//default

__constant float ZCoordinateCut=0.5f;	//default
__constant float InversePhiBinSize=20 / (2.0f * 3.14159265359f) ;
__constant float LayersZCoordinate[7]={16.333f, 16.333f, 16.333f, 42.140f, 42.140f, 73.745f, 73.745f};
__constant float LayersRCoordinate[7]={2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f, 34.388f, 39.3329f};
__constant float InverseZBinSize[7]=  {0.5f * 20 / 16.333f, 0.5f * 20 / 16.333f, 0.5f * 20 / 16.333f,0.5f * 20 / 42.140f, 0.5f * 20 / 42.140f, 0.5f * 20 / 73.745f, 0.5f * 20 / 73.745f };


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


Int4Struct getBinsRect(ClusterStruct* currentCluster, int layerIndex,float directionZIntersection)
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
/*
__kernel void countLayerTracklets(
				__global Float3Struct* primaryVertex,	//0
				__global ClusterStruct* currentLayerClusters, //1
				__global ClusterStruct* nextLayerClusters, //2
				__global int * currentLayerIndexTable, //3
				__global int * iCurrentLayer, //4
				__global int * iLayerClusterSize, //5
				__global int * iTrackletsPerClusterTablePreviousLayer //6
		)
					
{
	const int currentClusterIndex=get_global_id(0);
	int clusterTrackletsNum=0;

	int iLayer=*iCurrentLayer;
	
	
	int maxLayerCluster=iLayerClusterSize[iLayer];
	int currentLayerClusterVectorSize=iLayerClusterSize[iLayer];
	
	int nextLayerClusterVectorSize=iLayerClusterSize[iLayer+1];
	
	if(currentClusterIndex>=maxLayerCluster){
		return;
	}
	

	
	if(currentClusterIndex<currentLayerClusterVectorSize){
		ClusterStruct currentCluster=currentLayerClusters[currentClusterIndex];
		//printf("[%d] %d\t%d\n",currentClusterIndex,currentCluster.clusterId,currentCluster.indexTableBinIndex);
		float tanLambda=(currentCluster.zCoordinate-primaryVertex->z)/currentCluster.rCoordinate;
		float directionZIntersection= tanLambda*(LayersRCoordinate[iLayer+1]-currentCluster.rCoordinate)+currentCluster.zCoordinate;

		const Int4Struct selectedBinsRect=getBinsRect(&currentCluster,iLayer,directionZIntersection);
		if (selectedBinsRect.x != 0 || selectedBinsRect.y != 0 || selectedBinsRect.z != 0 || selectedBinsRect.w != 0) {
	      	const int nextLayerClustersNum=nextLayerClusterVectorSize;
	      	int phiBinsNum=selectedBinsRect.w - selectedBinsRect.y + 1;
			if(phiBinsNum<0){
		    	  phiBinsNum+=PhiBins;
		    }
			
		    for(int iPhiBin=selectedBinsRect.y,iPhiCount=0;iPhiCount < phiBinsNum;iPhiBin = ++iPhiBin == PhiBins ? 0 : iPhiBin, iPhiCount++){
		    	   
		    	  const int firstBinIndex=getBinIndex(selectedBinsRect.x,iPhiBin);
		    	  const int firstRowClusterIndex = currentLayerIndexTable[firstBinIndex];
		    	  const int maxRowClusterIndex = currentLayerIndexTable[firstBinIndex+ selectedBinsRect.z - selectedBinsRect.x + 1 ];
				  

				  for (int iNextLayerCluster=firstRowClusterIndex;iNextLayerCluster <= maxRowClusterIndex && iNextLayerCluster < nextLayerClustersNum; ++iNextLayerCluster) {
		    			if(iNextLayerCluster>=nextLayerClusterVectorSize)
		    				break;
		    		
		    		  ClusterStruct nextCluster=nextLayerClusters[iNextLayerCluster];
		    		 
		    		  const float deltaZ=fabs(tanLambda * (nextCluster.rCoordinate - currentCluster.rCoordinate) + currentCluster.zCoordinate - nextCluster.zCoordinate);
		    		  const float deltaPhi=fabs(currentCluster.phiCoordinate - nextCluster.phiCoordinate);

		    		  if (deltaZ < TrackletMaxDeltaZThreshold[iLayer] && (deltaPhi<PhiCoordinateCut || fabs(deltaPhi-TwoPi)<PhiCoordinateCut)){
		    			  ++clusterTrackletsNum;
		    			  
		    		  }
		    	  }
		      }
		}
	}	

	if(clusterTrackletsNum>0) {
		iTrackletsPerClusterTablePreviousLayer[currentClusterIndex] = clusterTrackletsNum;
  	}
  	else{
    	iTrackletsPerClusterTablePreviousLayer[currentClusterIndex] = 0;
  	}
}
*/

__kernel void countLayerTracklets(
				__global Float3Struct* primaryVertex,	//0
				__global ClusterStruct* currentLayerClusters, //1
				__global ClusterStruct* nextLayerClusters, //2
				__global int * currentLayerIndexTable, //3
				__global int * iCurrentLayer, //4
				__global int * iLayerClusterSize, //5
				__global int * iTrackletsPerClusterTablePreviousLayer, //6
				__global int * iCurrentTrackletsPosition, //7 
				__global TrackletStruct* currentLayerTracklets //8
		)
					
{
	const int currentClusterIndex=get_global_id(0);
	int clusterTrackletsNum=0;

	int iLayer=*iCurrentLayer;
	
	
	int maxLayerCluster=iLayerClusterSize[iLayer];
	int currentLayerClusterVectorSize=iLayerClusterSize[iLayer];
	
	int nextLayerClusterVectorSize=iLayerClusterSize[iLayer+1];
	
	if(currentClusterIndex>=maxLayerCluster){
		return;
	}
	

	
	if(currentClusterIndex<currentLayerClusterVectorSize){
		ClusterStruct currentCluster=currentLayerClusters[currentClusterIndex];
		//printf("[%d] %d\t%d\n",currentClusterIndex,currentCluster.clusterId,currentCluster.indexTableBinIndex);
		float tanLambda=(currentCluster.zCoordinate-primaryVertex->z)/currentCluster.rCoordinate;
		float directionZIntersection= tanLambda*(LayersRCoordinate[iLayer+1]-currentCluster.rCoordinate)+currentCluster.zCoordinate;

		const Int4Struct selectedBinsRect=getBinsRect(&currentCluster,iLayer,directionZIntersection);
		if (selectedBinsRect.x != 0 || selectedBinsRect.y != 0 || selectedBinsRect.z != 0 || selectedBinsRect.w != 0) {
	      	const int nextLayerClustersNum=nextLayerClusterVectorSize;
	      	int phiBinsNum=selectedBinsRect.w - selectedBinsRect.y + 1;
			if(phiBinsNum<0){
		    	  phiBinsNum+=PhiBins;
		    }
			
		    for(int iPhiBin=selectedBinsRect.y,iPhiCount=0;iPhiCount < phiBinsNum;iPhiBin = ++iPhiBin == PhiBins ? 0 : iPhiBin, iPhiCount++){
		    	   
		    	  const int firstBinIndex=getBinIndex(selectedBinsRect.x,iPhiBin);
		    	  const int firstRowClusterIndex = currentLayerIndexTable[firstBinIndex];
		    	  const int maxRowClusterIndex = currentLayerIndexTable[firstBinIndex+ selectedBinsRect.z - selectedBinsRect.x + 1 ];
				  

				  for (int iNextLayerCluster=firstRowClusterIndex;iNextLayerCluster <= maxRowClusterIndex && iNextLayerCluster < nextLayerClustersNum; ++iNextLayerCluster) {
		    			if(iNextLayerCluster>=nextLayerClusterVectorSize)
		    				break;
		    		
		    		  ClusterStruct nextCluster=nextLayerClusters[iNextLayerCluster];
		    		 
		    		  const float deltaZ=fabs(tanLambda * (nextCluster.rCoordinate - currentCluster.rCoordinate) + currentCluster.zCoordinate - nextCluster.zCoordinate);
		    		  const float deltaPhi=fabs(currentCluster.phiCoordinate - nextCluster.phiCoordinate);

		    		  if (deltaZ < TrackletMaxDeltaZThreshold[iLayer] && (deltaPhi<PhiCoordinateCut || fabs(deltaPhi-TwoPi)<PhiCoordinateCut)){
		    			  int iTrackletPosition=atom_inc(&iCurrentTrackletsPosition[iLayer]);
		    			  __global TrackletStruct* tracklet=&currentLayerTracklets[iTrackletPosition];
		    			  tracklet->firstClusterIndex=currentClusterIndex;
		    			  tracklet->secondClusterIndex=iNextLayerCluster;
		    			  tracklet->tanLambda=(currentCluster.zCoordinate - nextCluster.zCoordinate) / (currentCluster.rCoordinate - nextCluster.rCoordinate);
		    			  tracklet->phiCoordinate= atan2(currentCluster.yCoordinate - nextCluster.yCoordinate, currentCluster.xCoordinate - nextCluster.xCoordinate);
		    			  ++clusterTrackletsNum;
		    			  
		    		  }
		    	  }
		      }
		}
	}	

	if(clusterTrackletsNum>0) {
		iTrackletsPerClusterTablePreviousLayer[currentClusterIndex] = clusterTrackletsNum;
  	}
  	else{
    	iTrackletsPerClusterTablePreviousLayer[currentClusterIndex] = 0;
  	}
}

