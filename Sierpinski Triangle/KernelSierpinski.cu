#include "KernelSierpinski.cuh";

#include <cooperative_groups.h>;
namespace cg = cooperative_groups;

/*
* Kernel method for generating points for the Sierpinski Triangle.
* Each thread takes a triangle and finds three new points.
* 
* @points - device pointer for the buffer points that will be rendered from OpenGL.
* @triangle - initial triangle of the tree.
* @triangles - array that will be used to represent the triangles graph.
* @start_iteration - starting iteration that threads should start generetaing triangles.
* @max_iterations - max iteration 
* @threadShiftIndex - used for mapping threads to nodes in triangles graph.
*/
__global__ void kernel(float* points, Triangle* triangles, int start_iteration, int max_iteration, int threadShiftIndex) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	idx += threadShiftIndex;
	Point A;
	Point B;
	Point C;
	Point A1;
	Point B1;
	Point C1;
	Triangle triangle;
	Triangle t_1;
	Triangle t_2;
	Triangle t_3;


	auto g = cg::this_grid();


	if (idx == 0) {
		A.x = 0.0f;
		A.y = 1.29 - 0.4f;
		B.x = 0.75f;
		B.y = -0.4f;
		C.x = -0.75f;
		C.y = 0.0f - 0.4f;

		points[0] = A.x;  
		points[1] = A.y;  
		points[2] = B.x; 
		points[3] = B.y; 
		points[4] = C.x;  
		points[5] = C.y; 

		triangle.A = A;
		triangle.B = B;
		triangle.C = C;
		triangles[1] = triangle;
	}

	//

	__syncthreads();
	//Iterations start from 0
	for (int iteration = start_iteration; iteration <= max_iteration; iteration++) {
		int start_at = round(((pow(3, iteration) + 1))) / 2;
		int end_at = round(((pow(3, iteration+1) - 1))) / 2;

	
		if (idx >= start_at && idx <= end_at) {

			int parentNode = idx;
			triangle = triangles[parentNode];
			A = triangle.A;
			B = triangle.B;
			C = triangle.C;

			//DivideTriangle 
			A1.x = (A.x + B.x) / 2.0f;
			A1.y = (A.y + B.y) / 2.0f;
			B1.x = (B.x + C.x) / 2.0f;
			B1.y = (B.y + C.y) / 2.0f;
			C1.x = (C.x + A.x) / 2.0f;
			C1.y = (C.y + A.y) / 2.0f;

			//Make three new Triagnles
			t_1.A = A;
			t_1.B = A1;
			t_1.C = C1;

			t_2.A = A1;
			t_2.B = B;
			t_2.C = B1;

			t_3.A = C1;
			t_3.B = B1;
			t_3.C = C;

			//Insert three new triangles to triangles array
			triangles[3 * idx - 1] = t_1;
			triangles[3 * idx] = t_2;
			triangles[3 * idx + 1] = t_3;

			//Add three points 
			int offset = 2 * 3 * (idx);
			points[offset] = A1.x;
			points[offset + 1] = A1.y;
			points[offset + 2] = B1.x;
			points[offset + 3] = B1.y;
			points[offset + 4] = C1.x;
			points[offset + 5] = C1.y;
		}
		/*__syncthreads();*/
		g.sync();
	}


}

__global__ void divideTriangle(float* points, Triangle t, int iteration, int max_iterations, int id) {
	int idx = threadIdx.x;  // Since we are launching with 3 threads, this is sufficient

	/*int iteration;*/
	__shared__ Point A;
	__shared__ Point B;
	__shared__ Point C;
	__shared__ Point A1;
	__shared__ Point B1;
	__shared__ Point C1;

	A = t.A;
	B = t.B;
	C = t.C;
	Triangle smolTriangle;
	int iterations = 5;
	int numberOfThreadsNeededForIteration = 0;
	int numVertices;
	//Each thread generates 3 points;

	//ITERIMI I PARE
	if (iteration == 0) {
		if (idx == 0) {
			A.x = 0.0f;
			A.y = 1.29 - 0.4f;
			points[0] = A.x;  // x
			points[1] = A.y;  // y	
		}
		// First point
		if (idx == 1) {
			points[2] = 0.75f;  // x
			points[3] = 0.0f - 0.4f;  // y

			B.x = 0.75f;
			B.y = -0.4f;
		}

		if (idx == 2) {
			// Third point
			points[4] = -0.75f;  // x
			points[5] = 0.0f - 0.4f;  // y

			C.x = -0.75f;
			C.y = 0.0f - 0.4f;
		}
	}

	__syncthreads();

	int offset = 2 * 3 * id;
	//Mid point of AB
	if (idx == 0) {
		A1.x = (A.x + B.x) / 2.0f;
		A1.y = (A.y + B.y) / 2.0f;
		points[offset] = A1.x;
		points[offset + 1] = A1.y;
	}

	//Mid Point of BC
	if (idx == 1) {
		B1.x = (B.x + C.x) / 2.0f;
		B1.y = (B.y + C.y) / 2.0f;
		points[offset + 2] = B1.x;
		points[offset + 3] = B1.y;
	}

	//Mid Point of CA
	if (idx == 2) {
		C1.x = (C.x + A.x) / 2.0f;
		C1.y = (C.y + A.y) / 2.0f;
		points[offset + 4] = C1.x;
		points[offset + 5] = C1.y;
	}

	__syncthreads();

	//3 new triangles
	if (idx == 0) {
		smolTriangle.A = A;
		smolTriangle.B = A1;
		smolTriangle.C = C1;
	}

	if (idx == 1) {
		smolTriangle.A = A1;
		smolTriangle.B = B;
		smolTriangle.C = B1;
	}

	if (idx == 2) {
		smolTriangle.A = C1;
		smolTriangle.B = B1;
		smolTriangle.C = C;
	}

	iteration += 1;
	if (iteration <= max_iterations) {
		divideTriangle << <1, 3 >> > (points, smolTriangle, iteration, max_iterations, 3 * id - idx + 1);
	}


}


