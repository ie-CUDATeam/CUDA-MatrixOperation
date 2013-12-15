#include <iostream>

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>


int main(int argc, char *argv[])
{
    using namespace std;

    int devId = findCudaDevice(argc, (const char **)argv);
    cout << "devId = " << devId << endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devId);
    cout << "multiProcessorCount = " << prop.multiProcessorCount << endl;
    cout << "deviceOverlap = " << prop.deviceOverlap << endl;

    return 0;
}
