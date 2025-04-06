/*
 * Antenna Patterner:
 *     - Calculate the pattern from an input geometry + complex gain array or
 *     - Output a geometry to yield a specific input patter- Output a geometry
 *     to yield a specific input pattern.
 *
 * Temesgen Ataro. 2025.
 * WIP
 */

#include <stdio.h>
#define CUDA_CHECK(err) \
{ \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Cuda error: %s on line %d", cudaGetErrorString(err), __FILE__, __LINE__); \
    } \
}

int main()
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0))

    printf("===GPU Properties===\n");
    printf("%s\nTotal global memory: %zu GB\n", prop.name, prop.totalGlobalMem);
    return 0;
}
