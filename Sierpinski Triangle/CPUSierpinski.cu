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

#include "KernelSierpinski.cuh"
#include "SierpinskiTriangleVariables.cuh";

#include <chrono>
using namespace std;
using namespace std::chrono;

int t_size = pow(3, 19);
Triangle* triangles = new Triangle[t_size]; //4^12


void drawSierpinskiWithCPU() {
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

	A.x = 0.0f;
	A.y = 1.29 - 0.4f;
	B.x = 0.75f;
	B.y = -0.4f;
	C.x = -0.75f;
	C.y = 0.0f - 0.4f;

	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(0.29f, 0.44f, 0.55f);
	glBegin(GL_TRIANGLES);
	glVertex2f(A.x, A.y);
	glVertex2f(B.x, B.y);
	glVertex2f(C.x, C.y);
	glEnd();

	triangle.A = A;
	triangle.B = B;
	triangle.C = C;
	triangles[1] = triangle;

	glPointSize(1.0f);
	glColor3f(0.0f, 0.0f, 0.0f);

	//Iterations start from 0

	auto start = high_resolution_clock::now();
	int start_iteration = 0;
	int max_iteration = iterations;
	for (int iteration = start_iteration; iteration <= max_iteration; iteration++) {
		int start_at = round(((pow(3, iteration) + 1))) / 2;
		int end_at = round(((pow(3, iteration + 1) - 1))) / 2;
		for (int idx = start_at; idx <= end_at; idx++) {

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
			glBegin(GL_TRIANGLES);
			glVertex2f(A1.x, A1.y);
			glVertex2f(B1.x, B1.y);
			glVertex2f(C1.x, C1.y);
			glEnd();
		}
	}


	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	cout << "\nIterative negro Time taken by function: "
		<< duration.count() << " microseconds with iterations" <<iterations << endl;

}
