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
/// \file Constants.h
/// \brief 
///

#ifndef TRACKINGITSU_INCLUDE_CONSTANTS_H_
#define TRACKINGITSU_INCLUDE_CONSTANTS_H_

#ifndef __OPENCL_C_VERSION__
#include <climits>

#include "ITSReconstruction/CA/Definitions.h"

namespace o2
{
namespace ITS
{
namespace CA
{

namespace Constants {

namespace Math {
#endif

CONSTEXPR float Pi = 3.14159265359f ;
CONSTEXPR float TwoPi = 2.0f * 3.14159265359f ;
CONSTEXPR float FloatMinThreshold = 1e-20f ;

#ifndef __OPENCL_C_VERSION__
}

namespace ITS {

#endif
CONSTEXPR int LayersNumber = 7 ;
CONSTEXPR int TrackletsPerRoad = 6 ;
CONSTEXPR int CellsPerRoad = 5 ;
CONSTEXPR int UnusedIndex = -1 ;

//__constant float LayersZCoordinate[7]={16.333f, 16.333f, 16.333f, 42.140f, 42.140f, 73.745f, 73.745f};

#ifndef __OPENCL_C_VERSION__
GPU_HOST_DEVICE CONSTEXPR GPUArray<float, LayersNumber> LayersZCoordinate()
{
  return GPUArray<float, LayersNumber> {
#else
  __constant float LayersZCoordinate[7]=
#endif
	  { 16.333f, 16.333f, 16.333f, 42.140f, 42.140f, 73.745f, 73.745f }
#ifdef __OPENCL_C_VERSION__
  ;
#else
  };
}
#endif


#ifndef __OPENCL_C_VERSION__
GPU_HOST_DEVICE CONSTEXPR GPUArray<float, LayersNumber> LayersRCoordinate()
{
	return GPUArray<float, LayersNumber> {
#else
	__constant float LayersRCoordinate[7]=
#endif
	  { 2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f, 34.388f, 39.3329f }
#ifdef __OPENCL_C_VERSION__
  ;
#else
  };
}
}
#endif

#ifndef __OPENCL_C_VERSION__
namespace Thresholds {
GPU_DEVICE CONSTEXPR GPUArray<float, ITS::TrackletsPerRoad> TrackletMaxDeltaZThreshold()
{
	return GPUArray<float, ITS::TrackletsPerRoad> {
#else
	__constant float TrackletMaxDeltaZThreshold[6]=
#endif
	  { 0.1f, 0.1f, 0.3f, 0.3f, 0.3f, 0.3f }
#ifdef __OPENCL_C_VERSION__
  ;
#else
  };
}
#endif

CONSTEXPR float CellMaxDeltaTanLambdaThreshold = 0.025f ;


#ifndef __OPENCL_C_VERSION__
GPU_DEVICE CONSTEXPR GPUArray<float, ITS::CellsPerRoad> CellMaxDeltaZThreshold()
{
	return GPUArray<float, ITS::CellsPerRoad> {
#else
	__constant float CellMaxDeltaZThreshold[5]=
#endif
	  { 0.2f, 0.4f, 0.5f, 0.6f, 3.0f}
#ifdef __OPENCL_C_VERSION__
  ;
#else
  };
}
#endif


#ifndef __OPENCL_C_VERSION__
GPU_DEVICE CONSTEXPR GPUArray<float, ITS::CellsPerRoad> CellMaxDistanceOfClosestApproachThreshold()
{
	return GPUArray<float, ITS::CellsPerRoad> {
#else
	__constant float CellMaxDistanceOfClosestApproachThreshold[5]=
#endif
	  { 0.05f, 0.04f, 0.05f, 0.2f, 0.4f}
#ifdef __OPENCL_C_VERSION__
  ;
#else
  };
}
#endif





CONSTEXPR float CellMaxDeltaPhiThreshold = 0.14f ;
CONSTEXPR float ZCoordinateCut = 0.5f ;
CONSTEXPR float PhiCoordinateCut = 0.3f ;
CONSTEXPR int CellsMinLevel = 5 ;

#ifndef __OPENCL_C_VERSION__
CONSTEXPR GPUArray<float, ITS::CellsPerRoad - 1> NeighbourCellMaxNormalVectorsDelta {
    { 0.002f, 0.009f, 0.002f, 0.005f } };
CONSTEXPR GPUArray<float, ITS::CellsPerRoad - 1> NeighbourCellMaxCurvaturesDelta { { 0.008f, 0.0025f, 0.003f, 0.0035f } };
}

namespace IndexTable {
#endif
CONSTEXPR int ZBins = 20 ;
CONSTEXPR int PhiBins = 20 ;
//CONSTEXPR float InversePhiBinSize = Constants::IndexTable::PhiBins / Constants::Math::TwoPi ;
CONSTEXPR float InversePhiBinSize = 20 / (2.0f * 3.14159265359f) ;




#ifndef __OPENCL_C_VERSION__
GPU_HOST_DEVICE CONSTEXPR GPUArray<float, ITS::LayersNumber> InverseZBinSize()
{
	return GPUArray<float, ITS::LayersNumber> {
#else
	__constant float InverseZBinSize[7]=
#endif
	  { 0.5 * 20 / 16.333f, 0.5 * 20 / 16.333f, 0.5 * 20 / 16.333f,
		      0.5 * 20 / 42.140f, 0.5 * 20 / 42.140f, 0.5 * 20 / 73.745f, 0.5 * 20 / 73.745f}
#ifdef __OPENCL_C_VERSION__
  ;
#else
  };
}
#endif


#ifndef __OPENCL_C_VERSION__

}

namespace Memory {
CONSTEXPR GPUArray<float, ITS::TrackletsPerRoad> TrackletsMemoryCoefficients { { 0.0016353f, 0.0013627f, 0.000984f,
    0.00078135f, 0.00057934f, 0.00052217f } };
CONSTEXPR GPUArray<float, ITS::CellsPerRoad> CellsMemoryCoefficients { { 2.3208e-08f, 2.104e-08f, 1.6432e-08f,
    1.2412e-08f, 1.3543e-08f } };
}

namespace PDGCodes {
CONSTEXPR int PionCode { 211 };
}

}

}
}
}
#endif
#endif /* TRACKINGITSU_INCLUDE_CONSTANTS_H_ */
