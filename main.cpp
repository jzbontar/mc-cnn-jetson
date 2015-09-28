#include <stdio.h>

#include <zed/Camera.hpp>

#include "GL/freeglut.h"
 
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

extern void stereo_init(int, int);
extern void stereo_run(unsigned char *, unsigned char *, unsigned char *);
extern int downsample_factor;

using namespace sl::zed;

unsigned char *d_display;
GLuint tex_display;
cudaGraphicsResource* pcu_display;
Camera* zed;
int w, h, screenshot;

#define ok(expr) if (expr != 0) { printf("ERROR on line %d\n", __LINE__); exit(-1); }

void save_ppm(const char *fname, unsigned char *d_data, int width, int height)
{
    unsigned char *h_data = (unsigned char *)malloc(width * height * 4);
    cudaMemcpy(h_data, d_data, width * height * 4, cudaMemcpyDeviceToHost);
    FILE *f = fopen(fname, "w");
    fprintf(f, "P3 %d %d 255\n", width, height);
    for (int i = 0; i < width * height; i++) {
        unsigned char b = h_data[i * 4 + 0];
        unsigned char g = h_data[i * 4 + 1];
        unsigned char r = h_data[i * 4 + 2];
        fprintf(f, "%u %u %u\n", r, g, b);
    }
    fclose(f);
    free(h_data);
}

unsigned char *load_ppm(const char *fname)
{
    int width, height;

    FILE *f = fopen(fname, "r");
    int n = fscanf(f, "P3 %d %d 255\n", &width, &height);
    printf("image size: %d x %d\n", width, height);

    unsigned char *h_data = (unsigned char *)malloc(width * height * 4);
    for (int i = 0; i < height * width; i++) {
        int r, g, b;

        n = fscanf(f, "%d %d %d\n", &r, &g, &b);
        h_data[i * 4 + 0] = b;
        h_data[i * 4 + 1] = g;
        h_data[i * 4 + 2] = r;
        h_data[i * 4 + 3] = 0;
    }

    unsigned char *d_data;
    cudaMalloc((void **)&d_data, width * height * 4);
    cudaMemcpy(d_data, h_data, width * height * 4, cudaMemcpyHostToDevice);

    free(h_data);
    fclose(f);

    return d_data;
}


void keyboard(unsigned char key, int x, int y)
{
	switch(key) {
	case 's':
		screenshot = true;
		break;
	case 'q':
		glutDestroyWindow(1);
		break;
	}
}

void draw()
{
	if (zed->grab(SENSING_MODE::RAW, false, false) == 0) {
		Mat left = zed->retrieveImage_gpu(SIDE::LEFT);
		Mat right = zed->retrieveImage_gpu(SIDE::RIGHT);

		if (screenshot) {
			printf("screenshot\n");
			save_ppm("tmp/left.ppm", left.data, w, h);
			save_ppm("tmp/right.ppm", right.data, w, h);
			screenshot = 0;
		}

		stereo_run(left.data, right.data, d_display);

		cudaArray_t ArrIm;
		cudaGraphicsMapResources(1, &pcu_display, 0);
		cudaGraphicsSubResourceGetMappedArray(&ArrIm, pcu_display, 0, 0);
		cudaMemcpy2DToArray(ArrIm, 0, 0, d_display, 4 * (w / downsample_factor), 4 * (w / downsample_factor), 2 * (h / downsample_factor), cudaMemcpyDeviceToDevice);
		cudaGraphicsUnmapResources(1, &pcu_display, 0);

		glDrawBuffer(GL_BACK);
		glBindTexture(GL_TEXTURE_2D, tex_display);
		glBegin(GL_QUADS);
		glTexCoord2f(0.0,1.0);
		glVertex2f(-1.0,-1.0);
		glTexCoord2f(1.0,1.0);
		glVertex2f(1.0,-1.0);
		glTexCoord2f(1.0,0.0);
		glVertex2f(1.0,1.0);
		glTexCoord2f(0.0,0.0);
		glVertex2f(-1.0,1.0);
		glEnd();

		glutSwapBuffers();
	}
	glutPostRedisplay();
}


int main(int argc, char **argv) 
{
	if (argc == 1) {
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
		glutInitWindowPosition(50, 25);
		glutInitWindowSize(640, 720);
		glutCreateWindow("mccnn");
		
		zed = new Camera(sl::zed::HD720, 15.0);
		ok(zed->init(MODE::NONE, 0, true, false));

		w = zed->getImageSize().width;
		h = zed->getImageSize().height;

		glEnable(GL_TEXTURE_2D);	
		glGenTextures(1, &tex_display);
		glBindTexture(GL_TEXTURE_2D, tex_display);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, 4, (w / downsample_factor), 2 * (h / downsample_factor), 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);
		ok(cudaGraphicsGLRegisterImage(&pcu_display, tex_display, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));

		stereo_init(w, h);
		cudaMalloc(&d_display, (w / downsample_factor) * (h / downsample_factor) * 4 * 2);

		screenshot = 0;

		glutKeyboardFunc(keyboard);
		glutDisplayFunc(draw);
		glutMainLoop();
	} else {
		w = 1280;
		h = 720;

		unsigned char *d_left = load_ppm("tmp/left.ppm");
		unsigned char *d_right = load_ppm("tmp/right.ppm");

		stereo_init(w, h);
		cudaMalloc(&d_display, (w / downsample_factor) * (h / downsample_factor) * 4 * 2);
		stereo_run(d_left, d_right, d_display);
		save_ppm("tmp/out.ppm", d_display, (w / downsample_factor), (h / downsample_factor) * 2);
	}
	
	return 0;
}
