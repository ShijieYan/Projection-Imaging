#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "function.h"

using namespace std;

//set the dimensions
#define Nx 208.0f
#define Ny 256.0f
#define Nz 225.0f
#define h 50.0f
#define H 800.0f //H must be larger than h + Nz
//initially, the source is at Nx/2, Ny/2, h+Nz, change offset if you want to move the source horizontally
#define offset_x 200.0f 
#define offset_y 0.0f
//detector resolution, change if you want, but it will significantly slow down your simulation if it is too small
#define D 0.5f 

//kernel function, runs on GPU, inputs: 2d array that stores the centers of each pixel of the detector; 1d array to store the info of 5 faces of cuboid; Float3 pointer of position of source
									//  2d array to store the remaining weight (same dimension as pixel_centers); pointers store the dimension of two 2d arrays
__global__ void projection_imaging_kernel(Float3 **pixels_centers, Plane1d *walls, float *mus, Float3 *source, float **weight, int *rows, int *cols) {  
	float s_x = source->x;   //s_x stores x position of source
	float s_y = source->y;   //s_x stores y position of source
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	//indexing the local thread IDx and IDy in each 2d block specified by blockIdx and IDy
	if (row < *rows && col < *cols){ //if it is a valid thread
		Float3 &pixelcenter = pixels_centers[row][col]; //get position of corresponding pixel center 
		Float3 pin; //to store the incident position of the ray on the cuboid
		Float3 unitvector;   //to store unit direction vector, once source and pixel center set, the propagation direction is determined 
		int polar;  //to indicate the polarization of the plane where the intersection is, 0 means the plane is parallele to x, 1 means y, and 2 means z 
		//the upper bound is always the first to test(d = h + Nz)
		//there is always a result, even if it is not on the face, it will not affect the result
		Ray_plane_intersection(*source, pixelcenter, walls[0], &pin, &unitvector, &polar); 
		//fucntion Ray_plane_intersection will get the incident point (pin), unit direction vector and polarization of incident face correspond to each pixel center
		//it is very likely that the pin is on the upper bound, test it first for efficiency
		if (pin.z == walls[0].d && pin.x <= Nx && pin.x >= 0 && pin.y <= Ny && pin.y >= 0){
			goto jumpandsave; //go to cuboid tracer once a valid pin is calculated
		}
		//if not, it means pin is not on the upper face, test more faces, basically, there are other 8 circumstances, based on the relative position between source and (Nx,Ny)
		if (s_y >= 0 && s_y <= Ny) { 
			if (s_x > Nx) {
				Ray_plane_intersection(*source, pixelcenter, walls[4], &pin, &unitvector, &polar);
				goto jumpandsave; 
			}
			if (s_x < 0) {
				Ray_plane_intersection(*source, pixelcenter, walls[3], &pin, &unitvector, &polar);
				goto jumpandsave; 
			}	
		}

		if (s_x >= 0 && s_x <= Nx) {
			if (s_y > Ny) {
				Ray_plane_intersection(*source, pixelcenter, walls[2], &pin, &unitvector, &polar);
				goto jumpandsave; 
			}
			if (s_y < 0) {
				Ray_plane_intersection(*source, pixelcenter, walls[1], &pin, &unitvector, &polar);
				goto jumpandsave; 
			}	
		}

		if (s_x > Nx){
			if (s_y > Ny){
				Ray_plane_intersection(*source, pixelcenter, walls[2], &pin, &unitvector, &polar); 
				if (pin.y == walls[2].d && pin.x <= Nx && pin.x >= 0 && pin.z <= Nz + h && pin.z >= h){
					goto jumpandsave;
				} else {
					Ray_plane_intersection(*source, pixelcenter, walls[4], &pin, &unitvector, &polar);
					goto jumpandsave; 
				}
			}
		}

		if (s_x > Nx){
			if (s_y < 0){
				Ray_plane_intersection(*source, pixelcenter, walls[1], &pin, &unitvector, &polar); 
				if (pin.y == walls[1].d && pin.x <= Nx && pin.x >= 0 && pin.z <= Nz + h && pin.z >= h){
					goto jumpandsave;
				} else {
					Ray_plane_intersection(*source, pixelcenter, walls[4], &pin, &unitvector, &polar);
					goto jumpandsave; 
				}
			}
		}

		if (s_x < 0){
			if (s_y < 0){
				Ray_plane_intersection(*source, pixelcenter, walls[1], &pin, &unitvector, &polar); 
				if (pin.y == walls[1].d && pin.x <= Nx && pin.x >= 0 && pin.z <= Nz + h && pin.z >= h){
					goto jumpandsave;
				} else {
					Ray_plane_intersection(*source, pixelcenter, walls[3], &pin, &unitvector, &polar);
					goto jumpandsave; 
				}
			}
		}	

		if (s_x < 0){
			if (s_y > Ny){
				Ray_plane_intersection(*source, pixelcenter, walls[2], &pin, &unitvector, &polar); 
				if (pin.y == walls[2].d && pin.x <= Nx && pin.x >= 0 && pin.z <= Nz + h && pin.z >= h){
					goto jumpandsave;
				} else {
					Ray_plane_intersection(*source, pixelcenter, walls[3], &pin, &unitvector, &polar);
					goto jumpandsave; 
				}
			}
		}

		jumpandsave:
		//so far, we got the pin, unitvector, pin_plane
		float correction;
		//correct pin exactly the same way in onemove_in_cube just in case. e.g. pin.z = 15.00001 and unit vector is (0, 0 ,-1), this will cause problem
		switch (polar){
			case 0:
			correction = my_nextafterf((int)(pin.x + 0.5f), (unitvector.x > 0.f) - (unitvector.x < 0.f));
			pin.x = correction;
			case 1:
			correction = my_nextafterf((int)(pin.y + 0.5f), (unitvector.y > 0.f) - (unitvector.y < 0.f));
			pin.y = correction;
			case 2:
			correction = my_nextafterf((int)(pin.z + 0.5f), (unitvector.z > 0.f) - (unitvector.z < 0.f));
			pin.z = correction;
		}
		float w = 1.0f; //initial weight
		int ID; //indicator of intesection point's relationship with x,y and z plane
		float out_intersection[3];//store the output position
		float mu = 0.0f; //default, no attenuation
		Int3 pin_int;
		pin_int.ix = floorf(pin.x); //get the index of the voxel where the partical is right now
		pin_int.iy = floorf(pin.y);
		pin_int.iz = floorf(pin.z - h); //converted to global coordinate before getting the index in cuboid
		while( pin_int.ix >= 0 && pin_int.ix < Nx && pin_int.iy >= 0 && pin_int.iy < Ny && pin_int.iz >= 0 && pin_int.iz < Nz ){ //when the partical is still inside cuboid, cont. ray tracing
			float distance_onemove = onemove_in_cube(&pin, &unitvector, &out_intersection[0], &ID);
			//update next incident point pin
			pin.x = out_intersection[0];
			pin.y = out_intersection[1];
			pin.z = out_intersection[2];
			//convert float to int, get index of the present voxel
			pin_int.ix = floorf(pin.x);
			pin_int.iy = floorf(pin.y);
			pin_int.iz = floorf(pin.z - h);
			//fetch mu from *mus
			mu = *(mus + pin_int.iz * (int)Nx * (int)Ny + pin_int.iy * (int)Nx + pin_int.ix);
			w *= exp(- mu * distance_onemove); //update weight
		}
		//record the remaining weight
		weight[row][col] = w;
	}
}

int main(){
	//initialize source
	Float3 source;
	source.x = float(Nx)/2.0f + offset_x;
	source.y = float(Ny)/2.0f + offset_y;
	source.z = H;
	int Dim_total = Nx * Ny * Nz;
	//read in the data of cuboid, courtesy of Dr.Fang
	float *mus = (float*)malloc(Dim_total * sizeof(float));
	FILE *fp = fopen("headct.bin", "rb");
	fread(mus, sizeof(float), Dim_total, fp);
  	fclose(fp);
  	//convert the data to real mu, @50keV, 0.0193 mm-1, 0.0573 mm-1
  	float mubone = 0.0573;
  	float mufat = 0.0193;
  	for (int i = 0; i < Dim_total; i++){
  		if (mus[i] != 0){
  			float temp;
  			temp = (mus[i] - 0.3f)/(1.0f - 0.3f)*(mubone - mufat) + mufat;
  			mus[i] = temp;
  		} else {
  			mus[i] = 0.0f;
  		}
  	}
	//set up relevant planes
	Plane1d walls[6];
	//upper
	walls[0].d = h + Nz;
	walls[0].polar = 2;
	//front
	walls[1].d = 0;
	walls[1].polar = 1;
	//rear
	walls[2].d = Ny;
	walls[2].polar = 1;
	//left	
	walls[3].d = 0;
	walls[3].polar = 0;
	//right
 	walls[4].d = Nx;
	walls[4].polar = 0;
	//detector
	walls[5].d = 0;
	walls[5].polar = 2;
	//prepare the detector
	//Initialize the vertex of the cuboid
	Float3 vertex[8];
	vertex[0].x = 0;
	vertex[0].y = 0;
	vertex[0].z = h;
	vertex[1].x = Nx;
	vertex[1].y = 0;
	vertex[1].z = h;
	vertex[2].x = 0;
	vertex[2].y = Ny;
	vertex[2].z = h;
	vertex[3].x = Nx;
	vertex[3].y = Ny;
	vertex[3].z = h;
	vertex[4].x = 0;
	vertex[4].y = 0;
	vertex[4].z = h + Nz;
	vertex[5].x = Nx;
	vertex[5].y = 0;
	vertex[5].z = h + Nz;
	vertex[6].x = 0;
	vertex[6].y = Ny;
	vertex[6].z = h + Nz;
	vertex[7].x = Nx;
	vertex[7].y = Ny;
	vertex[7].z = h + Nz;
	//get the intesection of ray and detector to determine the dimension of detector
	Float3 detector_intersections[8], detector_unitvectors[8];
	int detector_polarizations[8];
	for (int i = 0; i < 8; i++){
		Ray_plane_intersection(source, vertex[i], walls[5], &detector_intersections[i], &detector_unitvectors[i], &detector_polarizations[i]);
	}
	//Now detector_intersections get the 8 intersection points, we only need to find maximum and minimum position in both x and y dimensions to determine the range of detector
	float detector_x[8] = {detector_intersections[0].x, detector_intersections[1].x, detector_intersections[2].x, detector_intersections[3].x, detector_intersections[4].x, detector_intersections[5].x, detector_intersections[6].x, detector_intersections[7].x};
	float detector_y[8] = {detector_intersections[0].y, detector_intersections[1].y, detector_intersections[2].y, detector_intersections[3].y, detector_intersections[4].y, detector_intersections[5].y, detector_intersections[6].y, detector_intersections[7].y};
	sort(&detector_x[0], &detector_x[0] + 8);
	sort(&detector_y[0], &detector_y[0] + 8);
	float minx = detector_x[0];
	float maxx = detector_x[7];
	float miny = detector_y[0];
	float maxy = detector_y[7];
	//we know the rectangular detector is specified by point (minx, miny, 0) and (maxx, maxy, 0)
	//calculate the dimension of detector based on the resolustion, you may notice, the dimension is increased by 2 to ensure each pixel that needed is contained in this detector
	const int Cols = ceil((maxx - minx)/D) + 2; //Mx dimension in x direction
	const int Rows = ceil((maxy - miny)/D) + 2; //My
	//allocate memory for input and output array
	float **h_weight = (float**)malloc(Rows * sizeof(float*));
	float *h_data_weight = (float*)malloc(Rows *Cols * sizeof(float));
	float **d_weight = NULL;  
	float *d_data_weight = NULL;
	//choose the GPU
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	//if memory allocation on GPU sucesss, it will print 'success' on terminal, if not, will print 'fail'
	cudaStatus = cudaMalloc((void**)&d_weight, Rows * sizeof(float*));
    if (cudaStatus == cudaSuccess) {
        cout<< "success" <<endl;
    }else{
    	cout<< "wrong" <<endl;
    }
	cudaStatus = cudaMalloc((void**)&d_data_weight, Rows * Cols * sizeof(float));
	   if (cudaStatus == cudaSuccess) {
        cout<< "success" <<endl;
    }else{
    	cout<< "wrong" <<endl;
    }
	//host and device pixelcenters allocation, weight for output allocated as well
	Float3 **h_pixels_centers = (Float3**)malloc(Rows * sizeof(Float3*));;
	Float3 *h_data_pixels_centers = (Float3*)malloc(Rows * Cols * sizeof(Float3));
	Float3 **d_pixels_centers = NULL;
	Float3 *d_data_pixels_centers = NULL;
	cudaStatus = cudaMalloc((void**)(&d_pixels_centers), Rows * sizeof(Float3*));
    if (cudaStatus == cudaSuccess) {
        cout<< "success" <<endl;
    }else{
    	cout<< "wrong" <<endl;
    }
	cudaStatus = cudaMalloc((void**)(&d_data_pixels_centers), Rows * Cols * sizeof(Float3));
    if (cudaStatus == cudaSuccess) {
        cout<< "success" <<endl;
    }else{
    	cout<< "wrong" <<endl;
    }
    //calculate the postion of each pixel
	for (int k = 0; k < Cols * Rows; k++){
		int i = k % Cols;
		int j = k / Cols;
		h_data_pixels_centers[k].x = minx - D/2 + D * i;
		h_data_pixels_centers[k].y = miny - D/2 + D * j;
		h_data_pixels_centers[k].z = 0;
	}
	//map the host 2d pointer to 1d data on the device for the convenience of introducing 2d array pointer on the device later
	for (int i = 0; i < Rows; i++){
		h_pixels_centers[i] = d_data_pixels_centers + i * Cols;
		h_weight[i] = d_data_weight + i * Cols;
	}
	//copy the pixel center data from host to device
	cudaStatus = cudaMemcpy((void*)d_data_pixels_centers, (void*)h_data_pixels_centers, Rows * Cols * sizeof(Float3), cudaMemcpyHostToDevice);
	if (cudaStatus == cudaSuccess) {
        cout<< "success" <<endl;
    }else{
    	cout<< "wrong" <<endl;
    }
	cudaStatus = cudaMemcpy((void*)d_pixels_centers, (void*)h_pixels_centers, Rows * sizeof(Float3*), cudaMemcpyHostToDevice);
	if (cudaStatus == cudaSuccess) {
        cout<< "success" <<endl;
    }else{
    	cout<< "wrong" <<endl;
    }
	cudaStatus = cudaMemcpy((void*)d_weight, (void*)h_weight, Rows * sizeof(float*), cudaMemcpyHostToDevice);
	if (cudaStatus == cudaSuccess) {
        cout<< "success" <<endl;
    }else{
    	cout<< "wrong" <<endl;
    }
	//copy data of walls to device
	Plane1d *d_walls;
	cudaStatus = cudaMalloc((void**)(&d_walls), 6 * sizeof(Plane1d));
	if (cudaStatus == cudaSuccess) {
        cout<< "success" <<endl;
    }else{
    	cout<< "wrong" <<endl;
    }
	cudaStatus = cudaMemcpy((void*)d_walls, (void*)walls, 6 * sizeof(Plane1d), cudaMemcpyHostToDevice);
	if (cudaStatus == cudaSuccess) {
        cout<< "success" <<endl;
    }else{
    	cout<< "wrong" <<endl;
    }
	//copy data of source to device
	Float3 *d_source;
	cudaStatus = cudaMalloc((void**)&d_source, sizeof(Float3));
	if (cudaStatus == cudaSuccess) {
        cout<< "success" <<endl;
    }else{
    	cout<< "wrong" <<endl;
    }
	cudaStatus = cudaMemcpy((void*)d_source, (void*)&source, sizeof(Float3), cudaMemcpyHostToDevice);
	if (cudaStatus == cudaSuccess) {
        cout<< "success" <<endl;
    }else{
    	cout<< "wrong" <<endl;
    }
	//copy data of mu to device as a 1D array
	float *d_mus;
	cudaStatus = cudaMalloc((void**)&d_mus, Dim_total * sizeof(float));
	if (cudaStatus == cudaSuccess) {
        cout<< "success" <<endl;
    }else{
    	cout<< "wrong" <<endl;
    }
	cudaStatus = cudaMemcpy((void*)d_mus, (void*)mus, Dim_total * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus == cudaSuccess) {
        cout<< "success" <<endl;
    }else{
    	cout<< "wrong" <<endl;
    }
    //copy the dimension of detector to device
	int *d_Rows, *d_Cols;
	cudaStatus = cudaMalloc((void**)&d_Rows, sizeof(int));
	if (cudaStatus == cudaSuccess) {
        cout<< "success" <<endl;
    }else{
    	cout<< "wrong" <<endl;
    }
	cudaStatus = cudaMalloc((void**)&d_Cols, sizeof(int));
	if (cudaStatus == cudaSuccess) {
        cout<< "success" <<endl;
    }else{
    	cout<< "wrong" <<endl;
    }
	cudaStatus = cudaMemcpy((void*)d_Rows, (void*)&Rows, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus == cudaSuccess) {
        cout<< "success" <<endl;
    }else{
    	cout<< "wrong" <<endl;
    }
	cudaStatus = cudaMemcpy((void*)d_Cols, (void*)&Cols, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus == cudaSuccess) {
        cout<< "success" <<endl;
    }else{
    	cout<< "wrong" <<endl;
    }
	//specified the dimension of each block
	dim3 threadPerBlock(32, 32);
	dim3 blockNumber( (Cols + threadPerBlock.x -1)/threadPerBlock.x,  (Rows + threadPerBlock.y -1)/threadPerBlock.y);
	//call the kernel to achieve parallel computation
	projection_imaging_kernel<<<blockNumber, threadPerBlock>>>(d_pixels_centers, d_walls, d_mus, d_source, d_weight, d_Rows, d_Cols);
	//let the CPU stream wait up for all GPU work is done
    cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus == cudaSuccess) {
        cout<< "success" << endl;
    }else{
    	cout << cudaStatus << endl;
    }
    //copy the result from device to host
	cudaStatus = cudaMemcpy((void*)h_data_weight, (void*)d_data_weight, Rows * Cols * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus == cudaSuccess) {
        cout<< "success" <<endl;
    }
	//output data to a .txt file
	ofstream myfile("Projection.txt");                         
	if (myfile.is_open()) {
		for (int i = 0; i < Rows * Cols; i++) {
			if (i%Cols == 0 && i != 0) {
				myfile << endl;
			}
			myfile << h_data_weight[i] << ' ';
		}
		myfile.close();
	}else {
		printf("Unable to open file");
	}
	//free all the memory
	cudaFree(d_weight);
	cudaFree(d_data_weight);
	cudaFree(d_pixels_centers);
	cudaFree(d_data_pixels_centers);
	cudaFree(d_walls);
	cudaFree(d_source);
	cudaFree(d_mus);
	cudaFree(d_Rows);
	cudaFree(d_Cols);
	free(mus);
	free(h_weight);
	free(h_data_weight);
	free(h_pixels_centers);
	free(h_data_pixels_centers);
	return 0;
}