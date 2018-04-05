/*
 * KernelHeader.h
 *
 *  Created on: 05 apr 2018
 *      Author: frank
 */

#ifndef INCLUDE_ITSRECONSTRUCTION_CA_GPU_KERNELHEADER_H_
#define INCLUDE_ITSRECONSTRUCTION_CA_GPU_KERNELHEADER_H_

#ifdef __OPENCL_C_VERSION__
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

	__constant int prova=100;
#endif


#endif /* INCLUDE_ITSRECONSTRUCTION_CA_GPU_KERNELHEADER_H_ */
