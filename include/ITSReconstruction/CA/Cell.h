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
/// \file Cell.h
/// \brief 
///

#ifndef TRACKINGITSU_INCLUDE_CACELL_H_
#define TRACKINGITSU_INCLUDE_CACELL_H_
#ifndef __OPENCL_C_VERSION__
#include <array>
#include <vector>

#include "ITSReconstruction/CA/Definitions.h"

#if TRACKINGITSU_OCL_MODE
#include "ITSReconstruction/CA/gpu/StructGPUPrimaryVertex.h"
#endif
namespace o2
{
namespace ITS
{
namespace CA
{

class Cell
  final
  {
    public:
	  GPU_DEVICE Cell() = default;
      GPU_DEVICE Cell(const int, const int, const int, const int, const int, const float3&, const float);

#if TRACKINGITSU_OCL_MODE
      GPU_DEVICE Cell(CellStruct&);
#endif

      int getFirstClusterIndex() const;
      int getSecondClusterIndex() const;
      int getThirdClusterIndex() const;
      GPU_HOST_DEVICE int getFirstTrackletIndex() const;
      int getSecondTrackletIndex() const;
      int getLevel() const;
      float getCurvature() const;
      const float3& getNormalVectorCoordinates() const;
      void setLevel(const int level);

    private:
#else
	  typedef struct{
#endif
      CONST int mFirstClusterIndex 		INITINT;
      CONST int mSecondClusterIndex 	INITINT;
      CONST int mThirdClusterIndex		INITINT;
      CONST int mFirstTrackletIndex 	INITINT;
      CONST int mSecondTrackletIndex 	INITINT;
      CONST FLOAT3 mNormalVectorCoordinates INITFLOAT3;
      CONST float mCurvature 			INITFLOAT;
      int mLevel INITINT;

	  }
#ifdef __OPENCL_C_VERSION__
	  Cell;
#else
	  ;

  inline int Cell::getFirstClusterIndex() const
  {
    return mFirstClusterIndex;
  }

  inline int Cell::getSecondClusterIndex() const
  {
    return mSecondClusterIndex;
  }

  inline int Cell::getThirdClusterIndex() const
  {
    return mThirdClusterIndex;
  }

  GPU_HOST_DEVICE inline int Cell::getFirstTrackletIndex() const
  {
    return mFirstTrackletIndex;
  }

  inline int Cell::getSecondTrackletIndex() const
  {
    return mSecondTrackletIndex;
  }

  inline int Cell::getLevel() const
  {
    return mLevel;
  }

  inline float Cell::getCurvature() const
  {
    return mCurvature;
  }

  inline const float3& Cell::getNormalVectorCoordinates() const
  {
    return mNormalVectorCoordinates;
  }

  inline void Cell::setLevel(const int level)
  {
    mLevel = level;
  }

}
}
}
#endif
#endif /* TRACKINGITSU_INCLUDE_CACELL_H_ */
