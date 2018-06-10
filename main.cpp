#include <ITSReconstruction/CA/Definitions.h>
#include <ITSReconstruction/CA/Event.h>
#include <ITSReconstruction/CA/IOUtils.h>
#include <ITSReconstruction/CA/Label.h>
#include <ITSReconstruction/CA/Road.h>
#include <ITSReconstruction/CA/Tracker.h>
#include <stddef.h>
#include <sys/time.h>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#define TIME_BENCHMARK

#if defined HAVE_VALGRIND
# include <valgrind/callgrind.h>
#endif

int maxEvent,minEvent;

#if TRACKINGITSU_GPU_MODE
# include "ITSReconstruction/CA/gpu/Utils.h"
#endif


using namespace o2::ITS::CA;

std::string getDirectory(const std::string& fname)
{
  size_t pos = fname.find_last_of("\\/");
  return (std::string::npos == pos) ? "" : fname.substr(0, pos + 1);
}
//#define N 1024*32
int main(int argc, char** argv)
{

#if 0

	///// vexcl
	
	compute::device device = compute::system::default_device();
    compute::context context(device);
    compute::command_queue queue(context, device);
	std::vector<int> h(N);       // Host vector.
	for(int i=0;i<N;i++)
		h[i]=rand() % 10;
	
	//for(int i=0;i<5;i++)
	//	std::cout<<"["<<i<<"]: "<<h[i]<<"\t";
	//std::cout<<std::endl;
	
	vex::vector<int> d({queue}, N);  // Device vector.
	vex::copy(h, d);    // Copy data from host to device.
   	std::chrono::time_point<std::chrono::system_clock> start, end;
	vex::inclusive_scan(d,d,0);
	start = std::chrono::system_clock::now();
	vex::inclusive_scan(d,d,0);
	end = std::chrono::system_clock::now();
    int elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
	std::cout<<N<<"\t"<<elapsed_seconds<<std::endl;
	vex::copy(d,h);
	

	//for(int i=0;i<5;i++)
	//	std::cout<<"["<<i<<"]: "<<h[i]<<"\t";
	//std::cout<<std::endl;
	
	///////////
	
	
	
	
	//////////boost
	/*
	compute::device device = compute::system::default_device();
    compute::context context(device);
    compute::command_queue queue(context, device);
	
	
	
	int host_data[N];
	for(int i=0;i<N;i++)
		host_data[i]=rand() % 10;
	
	compute::vector<int> device_vector(N, context);
	
	 // copy from host to device
    compute::copy(
        host_data, host_data + N, device_vector.begin(), queue
    );
	std::chrono::time_point<std::chrono::system_clock> start, end;
	compute::inclusive_scan(device_vector.begin(),
							device_vector.end(),
							device_vector.begin(),
							queue);
	queue.finish();
	start = std::chrono::system_clock::now();
	compute::inclusive_scan(device_vector.begin(),
							device_vector.end(),
							device_vector.begin(),
							queue);
	queue.finish();
	end = std::chrono::system_clock::now();
    int elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
	std::cout<<N<<"\t"<<elapsed_seconds<<std::endl;
	
    // create vector on host
    std::vector<int> host_vector(N);

    // copy data back to host
    compute::copy(
        device_vector.begin(), device_vector.end(), host_vector.begin(), queue
    );
	*/
	///////////////////
	
	
	
	
	return 1;
}
#else

#if TRACKINGITSU_CUDA_MODE
	std::cout<<">> CUDA"<<std::endl;
#elif TRACKINGITSU_OCL_MODE
	std::cout<<">> OpenCl"<<std::endl;
#else
	std::cout<<">> CPU"<<std::endl;
#endif



  if (argv[1] == NULL) {

    std::cerr << "Please, provide a data file." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string eventsFileName(argv[1]);
  std::string benchmarkFolderName = getDirectory(eventsFileName);
  std::vector<Event> events = IOUtils::loadEventData(eventsFileName);
  const int eventsNum = events.size();
  std::vector<std::unordered_map<int, Label>> labelsMap;
  bool createBenchmarkData = false;
  std::ofstream correctRoadsOutputStream;
  std::ofstream duplicateRoadsOutputStream;
  std::ofstream fakeRoadsOutputStream;

  int verticesNum = 0;
  for (int iEvent = 0; iEvent < eventsNum; ++iEvent) {

    verticesNum += events[iEvent].getPrimaryVerticesNum();
  }

  if (argv[2] != NULL) {

    std::string labelsFileName(argv[2]);

    createBenchmarkData = true;
    labelsMap = IOUtils::loadLabels(eventsNum, labelsFileName);

    correctRoadsOutputStream.open(benchmarkFolderName + "CorrectRoads.txt");
    duplicateRoadsOutputStream.open(benchmarkFolderName + "DuplicateRoads.txt");
    fakeRoadsOutputStream.open(benchmarkFolderName + "FakeRoads.txt");
  }

  //clock_t t1, t2;
  std::chrono::time_point<std::chrono::system_clock> start, end;

  float totalTime = 0.f, minTime = std::numeric_limits<float>::max(), maxTime = -1;
#if defined MEMORY_BENCHMARK
  std::ofstream memoryBenchmarkOutputStream;
  memoryBenchmarkOutputStream.open(benchmarkFolderName + "MemoryOccupancy.txt");
#elif defined TIME_BENCHMARK
  std::ofstream timeBenchmarkOutputStream;
  timeBenchmarkOutputStream.open(benchmarkFolderName + "TimeOccupancy.txt");
#endif

  // Prevent cold cache benchmark noise
  Tracker<TRACKINGITSU_GPU_MODE> tracker{};
  tracker.clustersToTracks(events[0]);

#if defined GPU_PROFILING_MODE
  Utils::Host::gpuStartProfiler();
#endif

  for (size_t iEvent = 0; iEvent < events.size(); ++iEvent) {

    Event& currentEvent = events[iEvent];
    std::cout << "Processing event " << iEvent + 1 << std::endl;
    start = std::chrono::system_clock::now();
    //t1 = clock();

#if defined HAVE_VALGRIND
    // Run callgrind with --collect-atstart=no
    CALLGRIND_TOGGLE_COLLECT;
#endif

    try {
#if defined(MEMORY_BENCHMARK)
      std::vector<std::vector<Road>> roads = tracker.clustersToTracksMemoryBenchmark(currentEvent, memoryBenchmarkOutputStream);
#elif defined(DEBUG)
      std::vector<std::vector<Road>> roads = tracker.clustersToTracksVerbose(currentEvent);
#elif defined TIME_BENCHMARK
      std::vector<std::vector<Road>> roads = tracker.clustersToTracksTimeBenchmark(currentEvent, timeBenchmarkOutputStream);
#else
      std::vector<std::vector<Road>> roads = tracker.clustersToTracks(currentEvent);
#endif

#if defined HAVE_VALGRIND
      CALLGRIND_TOGGLE_COLLECT;
#endif

      //t2 = clock();
      end = std::chrono::system_clock::now();
      int elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
//      const float diff = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);

      totalTime += elapsed_seconds;

      if (minTime > elapsed_seconds){
        minTime = elapsed_seconds;
        minEvent=iEvent + 1;
      }
      if (maxTime < elapsed_seconds){
        maxTime = elapsed_seconds;
        maxEvent= iEvent + 1;
      }

      for(int iVertex = 0; iVertex < currentEvent.getPrimaryVerticesNum(); ++iVertex) {

        std::cout << "Found " << roads[iVertex].size() << " roads for vertex " << iVertex + 1 << std::endl;
      }
      std::cout << "Event " << iEvent + 1 << " processed in: " << elapsed_seconds << "ms"<< std::endl;

      if(currentEvent.getPrimaryVerticesNum() > 1) {

        std::cout << "Vertex processing mean time: " << elapsed_seconds / currentEvent.getPrimaryVerticesNum() << "ms" << std::endl;
      }

      std::cout << std::endl;

      if (createBenchmarkData) {

        IOUtils::writeRoadsReport(correctRoadsOutputStream, duplicateRoadsOutputStream, fakeRoadsOutputStream, roads,
            labelsMap[iEvent]);
      }

    } catch (std::exception& e) {

      std::cout << e.what() << std::endl;
    }
  }

#if defined GPU_PROFILING_MODE
  Utils::Host::gpuStopProfiler();
#endif

  std::cout << std::endl;
  std::cout << "Avg time: " << totalTime / verticesNum << "ms" << std::endl;
  std::cout << "Min time: " << minTime << "ms\t[event #" <<minEvent << "]" << std::endl;
  std::cout << "Max time: " << maxTime << "ms\t[event #" <<maxEvent << "]" << std::endl;

  return 0;
}
#endif
