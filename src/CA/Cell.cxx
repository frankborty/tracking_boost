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
/// \file CACell.cxx
/// \brief 
///

#include "ITSReconstruction/CA/Cell.h"

namespace o2
{
namespace ITS
{
namespace CA
{

GPU_DEVICE Cell::Cell(const int firstClusterIndex, const int secondClusterIndex, const int thirdClusterIndex,
    const int firstTrackletIndex, const int secondTrackletIndex, const float3& normalVectorCoordinates,
    const float curvature)
    : mFirstClusterIndex { firstClusterIndex }, mSecondClusterIndex { secondClusterIndex }, mThirdClusterIndex {
        thirdClusterIndex }, mFirstTrackletIndex(firstTrackletIndex), mSecondTrackletIndex(secondTrackletIndex), mCurvature { curvature }, mLevel { 1 }
{
  this->mNormalVectorCoordinates[0]=normalVectorCoordinates.x;
  this->mNormalVectorCoordinates[1]=normalVectorCoordinates.y;
  this->mNormalVectorCoordinates[2]=normalVectorCoordinates.z;
}

GPU_DEVICE Cell::Cell()
    : mFirstClusterIndex { -1 }, mSecondClusterIndex { -1 }, mThirdClusterIndex {
        -1 }, mFirstTrackletIndex{-1}, mSecondTrackletIndex{-1}, mNormalVectorCoordinates{0.f,0.f,0.f}, mCurvature { 0.f }, mLevel { 1 }
{
  // Nothing to do
}

#if TRACKINGITSU_OCL_MODE
GPU_DEVICE Cell::Cell(CellStruct& cellStruct)
    : mFirstClusterIndex { cellStruct.mFirstClusterIndex }, mSecondClusterIndex { cellStruct.mSecondClusterIndex }, mThirdClusterIndex {
    	cellStruct.mThirdClusterIndex }, mFirstTrackletIndex(cellStruct.mFirstTrackletIndex), mSecondTrackletIndex(cellStruct.mSecondTrackletIndex), mNormalVectorCoordinates{
    			cellStruct.mNormalVectorCoordinates.x,cellStruct.mNormalVectorCoordinates.y,cellStruct.mNormalVectorCoordinates.z}, mCurvature { cellStruct.mCurvature }, mLevel { 1 }
{
  // Nothing to do
}
#endif

}
}
}
