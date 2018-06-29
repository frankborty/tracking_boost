#ifndef TRAKINGITSU_INCLUDE_STRUCT_GPU_PRIMARY_H_
#define TRAKINGITSU_INCLUDE_STRUCT_GPU_PRIMARY_H_

#include <array>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/Constants.h"



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
		const int mFirstClusterIndex;
		const int mSecondClusterIndex;
		const int mThirdClusterIndex;
		const int mFirstTrackletIndex;
		const int mSecondTrackletIndex;
		FLOAT3 mNormalVectorCoordinates;
		const float mCurvature;
		int mLevel;
	}CellStruct;

	typedef struct{
		int firstClusterIndex;
		int secondClusterIndex;
		float tanLambda;
		float phiCoordinate;
	}TrackletStruct;



#endif /* TRAKINGITSU_INCLUDE_STRUCT_GPU_PRIMARY_H_ */
