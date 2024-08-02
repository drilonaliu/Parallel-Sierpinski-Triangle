#include "SierpinskiRendering.cuh";
#include "KernelSierpinski.cuh";
#include "SierpinskiTriangleVariables.cuh";
#include "CPUSierpinski.cuh";

void clearBackground();
void generatePointsUsingGPU();
void renderTriangleFromBuffer();
void generatePointsUsing100GPU();

bool onlyOnce = true;
Triangle* d_triangles = 0;
int iterations = 10;


#include <chrono>
using namespace std;
using namespace std::chrono;

bool CPUImplementation = false;
bool GPUImplementationNormal = false;
bool GPUImplementationFast = true;
bool movingAroundWithMouse = false;
bool generatePoints = true;

//OpenGL display function
void draw_func(void) {
	clearBackground();
	if (GPUImplementationNormal) {
		if (generatePoints) {
			generatePointsUsingGPU();
		}
		renderTriangleFromBuffer();
	}
	if (GPUImplementationFast) {
		if (generatePoints) {
			generatePointsUsing100GPU();
		}
		renderTriangleFromBuffer();
	}
	else if (CPUImplementation) {
		drawSierpinskiWithCPU();
		glutSwapBuffers();
	}
}


void clearBackground() {
	glClearColor(1.0, 1.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
}

double log3(int x) {
	return log(x) / log(3);
}

void generatePointsUsingGPU() {
	// me e provu
	if (onlyOnce) {
		int numTotalTriangles = (pow(3, 18 + 1) - 1) / 2;
		cudaMalloc((void**)&d_triangles, numTotalTriangles * sizeof(Triangle));
		onlyOnce = false;
	}

	int threads = 1024;
	int blocks = 1;

	//Map Graphics resources
	float* devPtr;
	size_t size;
	cudaGraphicsMapResources(1, &resource, NULL);
	cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);


	int max_iterations = iterations;
	int numTriangles = pow(3, iterations + 1) / 2;
	int kernelCalls = numTriangles / (blocks * threads);
	int start_iteration = 1;

	auto durations = 0;
	auto start = high_resolution_clock::now();
	for (int k = 0; k <= kernelCalls; k++) {
		int threadShiftIndex = k * (blocks * threads);
		int previuos_iteration = start_iteration;
		start_iteration = (log3(2 * threadShiftIndex + 1));
		/*	printf("Start Iteration is %d", start_iteration);*/
		kernel << <1, 1024 >> > (devPtr, d_triangles, start_iteration, max_iterations, threadShiftIndex);
		cudaDeviceSynchronize();
	}

	cudaGraphicsUnmapResources(1, &resource, NULL);
}

/*
* Makes a call to kernel to generate fractal tree points.
* Each kernel call will use every thread in GPU, hence using
* 100% of GPU.
*/
void generatePointsUsing100GPU() {

	//Find the number of threads and blocks needed for cooperative groups
	int threads;
	int blocks;
	int maxNumberOfThreads;
	cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, kernel, 0, 0);
	maxNumberOfThreads = blocks * threads;

	//Allocate memory for graph tree branch in GPU
	if (onlyOnce) {
		int numTotalTriangles = (pow(3, 17 + 1) - 1) / 2;
		cudaMalloc((void**)&d_triangles, numTotalTriangles * sizeof(Triangle));
		onlyOnce = false;
	}

	//Map resource to OpenGL
	float* devPtr;
	size_t size;
	cudaGraphicsMapResources(1, &resource, NULL);
	cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);

	//Launching Kernels
	int max_iterations = iterations;
	int numTriangles = pow(3, iterations + 1) / 2;
	int kernelCalls = numTriangles / (blocks * threads);
	int start_iteration = 2;

	//Keep calling the kernel until we generate all the data
	for (int k = 0; k <= kernelCalls; k++) {
		int threadShiftIndex = k * (blocks * threads);
		int previuos_iteration = start_iteration;
		start_iteration = log3(2 * threadShiftIndex + 1);
		void* kernelArgs[] = { &devPtr, &d_triangles, &start_iteration, &max_iterations, &threadShiftIndex };
		cudaLaunchCooperativeKernel((void*)kernel, blocks, threads, kernelArgs, 0, 0);
	}

	//End of kernel calls
	cudaGraphicsUnmapResources(1, &resource, NULL);
}


void renderTriangleFromBuffer() {
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
	glEnableVertexAttribArray(0);

	//Render first blue triangle
	glColor3f(0.333f, 0.42f, 0.184f);
	glDrawArrays(GL_TRIANGLES, 0, 3);

	//Render blackTriangles
	int numberVerticesBlackTriangles = (pow(3, iterations + 2) - 3) / 2;
	glColor3f(0.0f, 0.0f, 0.0f);
	glDrawArrays(GL_TRIANGLES, 3, numberVerticesBlackTriangles);

	glutSwapBuffers();
}