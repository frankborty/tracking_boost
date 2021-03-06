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
/// \file IOUtils.h
/// \brief 
///
/// \author Iacopo Colonnelli, Politecnico di Torino
///

#ifndef TRACKINGITSU_INCLUDE_EVENTLOADER_H_
#define TRACKINGITSU_INCLUDE_EVENTLOADER_H_

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "ITSReconstruction/CA/Event.h"
#include "ITSReconstruction/CA/Label.h"
#include "ITSReconstruction/CA/Road.h"

namespace o2
{
namespace ITS
{
namespace CA
{

namespace IOUtils {
std::vector<Event> loadEventData(const std::string&);
std::vector<std::unordered_map<int, Label>> loadLabels(const int, const std::string&);
void writeRoadsReport(std::ofstream&, std::ofstream&, std::ofstream&, const std::vector<std::vector<Road>>&,
    const std::unordered_map<int, Label>&);
}

}
}
}

#endif /* TRACKINGITSU_INCLUDE_EVENTLOADER_H_ */
