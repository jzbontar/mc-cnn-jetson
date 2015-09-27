#include <stdio.h>

#define TB 128

#define ok(expr) if (expr != 0) { printf("ERROR on line %d\n", __LINE__); exit(-1); }

float *left_gray, *right_gray;
int width, height, size;

void stereo_init(int width_arg, int height_arg)
{
	width = width_arg;
	height = height_arg;
	size = width * height;
	printf("stereo_init: %d x %d\n", width, height);

	ok(cudaMalloc((void **)&left_gray, size * 4));
	ok(cudaMalloc((void **)&right_gray, size * 4));
}

void __global__ rgb2gray(unsigned char *input, float *output, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		float b = input[id * 4 + 0];
		float g = input[id * 4 + 1];
		float r = input[id * 4 + 2];
		output[id] = 0.299 * r + 0.587 * g + 0.114 * b;
	}
}

void __global__ gray2display(float *input, unsigned char *display, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		display[id * 4 + 0] = input[id];
		display[id * 4 + 1] = input[id];
		display[id * 4 + 2] = input[id];
		display[id * 4 + 3] = 0;
	}
}

void stereo_run(unsigned char *left, unsigned char *right, unsigned char *display)
{
	rgb2gray<<<(size - 1) / TB + 1, TB>>>(left, left_gray, size);
	rgb2gray<<<(size - 1) / TB + 1, TB>>>(right, right_gray, size);

	gray2display<<<(size - 1) / TB + 1, TB>>>(left_gray, display, size);
	gray2display<<<(size - 1) / TB + 1, TB>>>(right_gray, display + size * 4, size);
}
