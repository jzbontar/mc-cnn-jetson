#include <stdio.h>

#include <zed/Camera.hpp>

#include "GL/freeglut.h"
 
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

using namespace sl::zed;

GLuint texLeft;
cudaGraphicsResource* pcuLeft;
Camera* zed;
int w, h;

#define ok(expr) if (expr != 0) { exit(-1); }

void keyboard(unsigned char key, int x, int y)
{
	switch(key) {
	case 'q':
		glutDestroyWindow(1);
		break;
	}
}

void draw()
{
	if (zed->grab(SENSING_MODE::RAW, false, false) == 0) {
		Mat left = zed->retrieveImage_gpu(SIDE::LEFT);
		cudaArray_t ArrIm;
		cudaGraphicsMapResources(1, &pcuLeft, 0);
		cudaGraphicsSubResourceGetMappedArray(&ArrIm, pcuLeft, 0, 0);
		cudaMemcpy2DToArray(ArrIm, 0, 0, left.data, left.step, w * 4, h, cudaMemcpyDeviceToDevice);
		cudaGraphicsUnmapResources(1, &pcuLeft, 0);

		glDrawBuffer(GL_BACK);
		glBindTexture(GL_TEXTURE_2D, texLeft);
		glBegin(GL_QUADS);
		glTexCoord2f(0.0,1.0);
		glVertex2f(-1.0,-1.0);
		glTexCoord2f(1.0,1.0);
		glVertex2f(0.0,-1.0);
		glTexCoord2f(1.0,0.0);
		glVertex2f(0.0,1.0);
		glTexCoord2f(0.0,0.0);
		glVertex2f(-1.0,1.0);
		glEnd();

		glutSwapBuffers();
	}
	glutPostRedisplay();
}


int main(int argc, char **argv) 
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowPosition(50, 25);
	glutInitWindowSize(1280, 480);
	glutCreateWindow("mccnn");
	
	zed = new Camera(sl::zed::HD720, 15.0);
	ok(zed->init(MODE::NONE, 0, true, false));

	w = zed->getImageSize().width;
	h = zed->getImageSize().height;

	glEnable(GL_TEXTURE_2D);	
	glGenTextures(1, &texLeft);
	glBindTexture(GL_TEXTURE_2D, texLeft);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, 4, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	ok(cudaGraphicsGLRegisterImage(&pcuLeft, texLeft, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));

	glutKeyboardFunc(keyboard);
	glutDisplayFunc(draw);
	glutMainLoop();
	
	return 0;
}
