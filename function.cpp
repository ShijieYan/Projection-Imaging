#include <math.h>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <string.h>
#include "function.h"

__device__ float my_nextafterf(float a, int dir) {
	union {
		float f;
		uint32_t i;
	}num;
	num.f = a + 1000.f;
	num.i += dir ^ (num.i & 0x80000000);
	return num.f - 1000.f;
}

__device__ float onemove_in_cube(Float3 *p0, Float3 *v, float *htime, int *id) {
	float dist, xi[3];

	//time-of-fly to hit the wall in each direction
	htime[0] = fabs((floorf(p0->x) + (v->x > 0.f) - p0->x) / v->x);
	htime[1] = fabs((floorf(p0->y) + (v->y > 0.f) - p0->y) / v->y);
	htime[2] = fabs((floorf(p0->z) + (v->z > 0.f) - p0->z) / v->z);

	//get the direction with the smallest time-of-fly
	dist = fmin(fmin(htime[0], htime[1]), htime[2]);
	(*id) = (dist == htime[0] ? 0 : (dist == htime[1] ? 1 : 2));

	//p0 is inside, p is outside, move to the 1st intersection, now in the air side, to be corrected in the else block
	htime[0] = p0->x + dist*v->x;
	htime[1] = p0->y + dist*v->y;
	htime[2] = p0->z + dist*v->z;
	xi[0] = my_nextafterf((int)(htime[0] + 0.5f), (v->x > 0.f) - (v->x < 0.f));
	xi[1] = my_nextafterf((int)(htime[1] + 0.5f), (v->y > 0.f) - (v->y < 0.f));
	xi[2] = my_nextafterf((int)(htime[2] + 0.5f), (v->z > 0.f) - (v->z < 0.f));
	if (*id == 0)htime[0] = xi[0];
	if (*id == 1)htime[1] = xi[1];
	if (*id == 2)htime[2] = xi[2];
	return dist;
}

__host__ __device__ void Ray_plane_intersection(const Float3 start, const Float3 end, const Plane1d wall, Float3 *intersection, Float3 *unitvector, int *polarization) { 
	float distance = wall.d;
	*polarization = wall.polar;
	//Initialize the unit vector for propagation
	Float3 direction;
	direction.x = end.x - start.x;
	direction.y = end.y - start.y;
	direction.z = end.z - start.z;
	//calculate the length of direction vector
	float length = sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
	float invert_length = 1.0f/length;
	//normalized to unit vector
	unitvector->x = direction.x * invert_length;
	unitvector->y = direction.y * invert_length;
	unitvector->z = direction.z * invert_length;
	//Initialize the position of the input point on the faces of cuboid
	float t;
	switch(*polarization){
		case 2:
		if(end.z != start.z){
			//calculate the intersection point of the ray and corresponding plane specified by *polarization
			intersection->z = distance;
			t = (distance - start.z)/(end.z - start.z);
			intersection->x = start.x + (end.x - start.x) * t;
			intersection->y = start.y + (end.y - start.y) * t;
		} 
		break;
		case 0:
		if(end.x != start.x){
			intersection->x = distance;
			t = (distance - start.x)/(end.x - start.x);
			intersection->y = start.y + (end.y - start.y) * t;
			intersection->z = start.z + (end.z - start.z) * t;
		} 
		break;
		case 1:
		if(end.y != start.y){
			intersection->y = distance;
			t = (distance - start.y)/(end.y - start.y);
			intersection->x = start.x + (end.x - start.x) * t;
			intersection->z = start.z + (end.z - start.z) * t;
		} 
		break;
		default: printf("polarization of the plane not given or wrong");
		break;
	}
}