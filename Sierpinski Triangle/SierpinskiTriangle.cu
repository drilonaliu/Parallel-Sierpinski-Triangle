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
//Sierpinski Triangle includes
#include "KernelSierpinski.cuh";
#include "SierpinskiTriangleVariables.cuh";
#include "SierpinskiRendering.cuh";
#include "UserInteraction.cuh";
#include "CPUSierpinski.cuh";

using namespace std;

GLuint bufferObj;
cudaGraphicsResource* resource;
int iteration = 0;
int numVertices = 3 * pow(4, iteration);
int max_iterations = 4;

void initializeWindow(int argc, char** argv);
void bindFunctionsToWindow();
void setUpCudaOpenGLInterop();

void startSierpinskiTriangle(int argc, char** argv) {
	initializeWindow(argc, argv);
	createMenu();
	setUpCudaOpenGLInterop();
	bindFunctionsToWindow();
	glutMainLoop();
}

void initializeWindow(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(512, 512);
	glutCreateWindow("Sierpinski Triangle");
	glewInit();
}

void bindFunctionsToWindow() {
	glutSpecialFunc(specialKeyHandler);
	glutKeyboardFunc(keyboardHandler);
	glutMouseFunc(mouseButton);
	glutMotionFunc(mouseMove);
	glutMouseWheelFunc(mouseWheel);
//	glutReshapeFunc(adjustCoordinateSystemOnResize);
	glutDisplayFunc(draw_func);
}

void setUpCudaOpenGLInterop() {
	//Choose the most suitable CUDA device based on the specified properties (in prop). It assigns the device ID to dev.
	cudaDeviceProp prop;
	int dev;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	cudaError_t error = cudaChooseDevice(&dev, &prop);
	if (error != cudaSuccess) {
		printf("Error choosing CUDA device: %s\n", cudaGetErrorString(error));
	}
	cudaGLSetGLDevice(dev);

	//Buffer Size
	int iterations = 4;
	int numVertices = 3 * pow(3, 15);
	size_t bufferSize = sizeof(float) * numVertices * 2;

	//Generate 1 buffer with id bufferObj
	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_ARRAY_BUFFER, bufferObj);
	glBufferData(GL_ARRAY_BUFFER, bufferSize, NULL, GL_DYNAMIC_COPY);

	//Notify CUDA runtime that we intend to share the OpenGL buffer named bufferObj with CUDA.//FlagsNone, ReadOnly, WriteOnly
	cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
}



