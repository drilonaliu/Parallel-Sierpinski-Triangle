// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>

// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <helper_functions.h>
#include <rendercheck_gl.h>
#include <device_launch_parameters.h>

__device__ struct Point {
	float x;
	float y;
};

__device__ struct Triangle {
	Point A;
	Point B;
	Point C;
};

__device__ void triangleOnSegment(Point A, Point B, Point* A1, Point* B1, Point* C1);
__global__ void divideTriangle(float* points, Triangle t, int iteration, int max_iterations, int id);
__global__ void kernel(float* points, Triangle* triangles, int start_iteration, int max_iteration, int threadShiftIndex);