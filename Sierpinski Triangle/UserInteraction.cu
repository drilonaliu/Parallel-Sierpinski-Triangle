// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>

#include "UserInteraction.cuh";
#include "SierpinskiTriangleVariables.cuh";


enum {
	MENU_OPTION_1,
	MENU_OPTION_2,
	MENU_OPTION_3,
	MENU_OPTION_4,
	MENU_EXIT
};

float zoomLevel = 1.0f;
float offsetX = 0.0f, offsetY = 0.0f;
float lastMouseX, lastMouseY;
bool isPanning = false;

void adjustCoordinateSystemOnResize(int width, int height) {
	glViewport(0, 0, width, height);
	float aspect = (1.0f * width) / (1.0f * height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(-1.6 * aspect, 1.6 * aspect, -1.7, 1.5);
	glutPostRedisplay();
}
void updateProjection() {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	int windowWidth = glutGet(GLUT_WINDOW_WIDTH);
	int windowHeight = glutGet(GLUT_WINDOW_HEIGHT);
	float aspectRatio = windowWidth/windowHeight; // Adjust this if your window has a different aspect ratio
	gluOrtho2D(-aspectRatio / zoomLevel + offsetX, aspectRatio / zoomLevel + offsetX, -1.0 / zoomLevel + offsetY, 1.0 / zoomLevel + offsetY);
	glMatrixMode(GL_MODELVIEW);
}

void mouseButton(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON) {
		if (state == GLUT_DOWN) {
			isPanning = true;
			movingAroundWithMouse = true;
			lastMouseX = x;
			lastMouseY = y;
		}
		else if (state == GLUT_UP) {
			isPanning = false;
		}
	}
}


void mouseWheel(int button, int dir, int x, int y) {
	if (dir > 0) {
		zoomLevel *= 1.1f; // Zoom in
	}
	else {
		zoomLevel /= 1.1f; // Zoom out
	}
	generatePoints = false;
	updateProjection();
	glutPostRedisplay(); // Redraw the scene
}

void mouseMove(int x, int y) {
	if (isPanning) {
		generatePoints = false;
		// Get the window dimensions
		int imageW = glutGet(GLUT_WINDOW_WIDTH);
		int imageH = glutGet(GLUT_WINDOW_HEIGHT);

		double fx = (double)(x - lastMouseX) / 50.0 / (double)(imageW);
		double fy = (double)(lastMouseY - y) / 50.0 / (double)(imageH);

		// Calculate the difference in mouse movement
		float dx = (float)(x - lastMouseX) / imageW;
		float dy = (float)(lastMouseY - y) / imageH; // Note: Y is inverted

		// Scale the movement by the zoom level and the orthographic projection extents
		float orthoWidth = 2.0f / zoomLevel; // Assuming initial projection is -1.0 to 1.0
		float orthoHeight = 2.0f / zoomLevel; // Assuming initial projection is -1.0 to 1.0

		offsetX += dx * orthoWidth;
		offsetY += dy * orthoHeight;


		// Update last mouse position
		lastMouseX = x;
		lastMouseY = y;

		updateProjection();
		glutPostRedisplay();
		//glutSwapBuffers();
	}
}

void keyboardHandler(unsigned char key, int x, int y) {
	switch (key) {
	case 'a':
	case 'A':
		zoomLevel *= 1.1f;
		updateProjection();
		//leftAngle = true;
		break;
	case 'd':
	case 'D':
		zoomLevel /= 1.1f;
		updateProjection();
		//leftAngle = false;
		break;
	case 's':
	case 'S':
		//rightAngle = true;
		//leftAngle = true;
		break;
	case 'r':
	case 'R':
		//rightAngle = true;
		//leftAngle = true;
		//angle_left = 3.14f / 6.0f;
		//angle_right = -3.14f / 6.0f;
		break;
	case 'f':
	case 'F':
		//	animation = !animation;
		break;
	case 'C':
	case 'c':
		//		color = !color;
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

void specialKeyHandler(int key, int x, int y) {
	switch (key) {
	case GLUT_KEY_LEFT:
		break;
	case GLUT_KEY_RIGHT:
		break;
	case GLUT_KEY_UP:
		iterations += 1;
		generatePoints = true;
		printf("Iterations: %d \r", iterations);
		break;
	case GLUT_KEY_DOWN:
		iterations -= 1;
		generatePoints = false;
		printf("Iterations: %d \r", iterations);
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

void handleMenu(int option) {
	switch (option) {
	case MENU_OPTION_1:
		CPUImplementation = true;
		GPUImplementationFast = false;
		GPUImplementationNormal = false;
		/*	GPUImplementationRecursive = false;*/
		break;
	case MENU_OPTION_2:
		CPUImplementation = false;
		GPUImplementationFast = true;
		GPUImplementationNormal = false;
		//	GPUImplementationRecursive = false;
		break;
	case MENU_OPTION_3:
		CPUImplementation = false;
		GPUImplementationFast = false;
		GPUImplementationNormal = true;
		//	GPUImplementationRecursive = false;
		break;
	case MENU_OPTION_4:
		CPUImplementation = false;
		GPUImplementationFast = false;
		GPUImplementationNormal = false;
		//	GPUImplementationRecursive = true;
		break;
	case MENU_EXIT:
		exit(0);
		break;
	}
	glutPostRedisplay();
}




void createMenu() {
	int menu = glutCreateMenu(handleMenu);
	glutAddMenuEntry("Switch to CPU implementation", MENU_OPTION_1);
	glutAddMenuEntry("Switch to fast parallel implementation (100% GPU usage)", MENU_OPTION_2);
	glutAddMenuEntry("Switch to parallel implementation (Normal GPU usage)", MENU_OPTION_3);
	glutAddMenuEntry("Switch to recursive parallel implementation (slow)", MENU_OPTION_4);
	glutAddMenuEntry("Exit", MENU_EXIT);
	glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void printControls() {
	printf("\nRight click on window to switch between different CPU and GPU implementations.");
	printf("\nPress [R] to reset ");
	printf("\nPress [arrowUp] to increase the iteration");
	printf("\nPress [arrowDown] to decrease the iteration");
	printf("\n\n");
}


//void motionFunc(int x, int y) {
//	double fx = (double)(x - lastx) / 50.0 / (double)(imageW);
//	double fy = (double)(lasty - y) / 50.0 / (double)(imageH);
//
//	if (leftClicked) {
//		xdOff = fx * scale;
//		ydOff = fy * scale;
//	}
//	else {
//		xdOff = 0.0f;
//		ydOff = 0.0f;
//	}
//
//	if (middleClicked)
//		if (fy > 0.0f) {
//			dscale = 1.0 - fy;
//			dscale = dscale < 1.05 ? dscale : 1.05;
//		}
//		else {
//			dscale = 1.0 / (1.0 + fy);
//			dscale = dscale > (1.0 / 1.05) ? dscale : (1.0 / 1.05);
//		}
//	else {
//		dscale = 1.0;
//	}
//}