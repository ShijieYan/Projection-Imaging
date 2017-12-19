#ifndef __dataStruc_function_h__
#define __dataStruc_function_h__
#include <cuda.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

typedef struct __align__(16) Float3 {
	float x;
	float y;
	float z;
}Float3;

typedef struct __align__(16) Int3 {
	int ix;
	int iy;
	int iz;
}Int3;

typedef struct __align__(16) Plane1d {
	float d;
	int polar;
}Plane1d;
//correct the incident position
__device__ float my_nextafterf(float a, int dir);
//ray tracer in one voxel
__device__ float onemove_in_cube(Float3 *p0, Float3 *v, float *htime, int *id);
//calculate the intersection b/w a ray and a plane
__host__ __device__ void Ray_plane_intersection(const Float3 start, const Float3 end, const Plane1d wall, Float3 *intersection, Float3 *unitvector, int *polarization);



#endif