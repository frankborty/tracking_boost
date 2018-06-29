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
/// \file Tracklet.h
/// \brief 
///




#ifndef TRACKINGITSU_INCLUDE_TRACKLET_H_
#define TRACKINGITSU_INCLUDE_TRACKLET_H_

#ifndef __OPENCL_C_VERSION__
#include "ITSReconstruction/CA/Cluster.h"

namespace o2
{
namespace ITS
{
namespace CA
{


struct Tracklet
	final
    {
      Tracklet();
      GPU_DEVICE Tracklet(const int, const int, const Cluster&, const Cluster&);

#else
      typedef struct{
#endif

      CONST int firstClusterIndex;
      CONST int secondClusterIndex;
      CONST float tanLambda;
      CONST float phiCoordinate;
  }
#ifdef __OPENCL_C_VERSION__
      Tracklet;
#else
	  ;

}
}
}
#endif
#endif /* TRACKINGITSU_INCLUDE_TRACKLET_H_ */
