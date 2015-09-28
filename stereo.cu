#include <stdio.h>
#include <assert.h>
#include <math_constants.h>

#define TB 128
#define GS(x) (((x) - 1) / TB + 1)

#define ok(expr) if (expr != 0) { printf("ERROR on line %d\n", __LINE__); exit(-1); }

__global__ void zero_(float *input, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
		input[id] = 0;
	}
}

__global__ void mul_(float *input, float factor, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
		input[id] = input[id] * factor;
	}
}

void __global__ rgb2gray_(unsigned char *input, float *output, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		float b = input[id * 4 + 0];
		float g = input[id * 4 + 1];
		float r = input[id * 4 + 2];
		output[id] = 0.299 * r + 0.587 * g + 0.114 * b;
	}
}

void __global__ gray2display_(float *input, unsigned char *display, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		display[id * 4 + 0] = input[id];
		display[id * 4 + 1] = input[id];
		display[id * 4 + 2] = input[id];
		display[id * 4 + 3] = 0;
	}
}

__global__ void ad_(float *x0, float *x1, float *output, int size, int size2, int size3, int direction)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < size) {
        int d = id;
        int x = d % size3;
        d /= size3;
        int y = d % size2;
        d /= size2;
        d *= direction;

        float dist;
        if (0 <= x + d && x + d < size3) {
            int cnt = 0;
            dist = 0;
            for (int yy = y - 2; yy <= y + 2; yy++) {
                for (int xx = x - 2; xx <= x + 2; xx++) {
                    if (0 <= xx && xx < size3 && 0 <= xx + d && xx + d < size3 && 0 <= yy && yy < size2) {
                        int ind = yy * size3 + xx;
                        dist += abs(x0[ind] - x1[ind + d]);
                        cnt++;
                    }
                }
            }
            dist /= cnt;
        } else {
            dist = CUDART_NAN;
        }
        output[id] = dist;
    }
}

__global__ void argmin_(float *input, float *output, int size, int size1, int size23)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        int argmin = 0;
        float min = CUDART_INF;
        for (int i = 0; i < size1; i++) {
            float val = input[i * size23 + id];
            if (val < min) {
                min = val;
                argmin = i;
            }
        }
        output[id] = argmin;
    }
}

__global__ void downsample_(float *input, float *output, int factor, int size3, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
		int dim3 = id % size3;
		int dim2 = id / size3;
		atomicAdd(output + ((dim2 / factor) * (size3 / factor) + (dim3 / factor)), input[id] / (factor * factor));
	}
}

void downsample(float *input, float *output, int factor, int size2_input, int size3_input)
{
	assert(size2_input % factor == 0);
	assert(size3_input % factor == 0);

	int size_input = size2_input * size3_input;
	int size_output = (size2_input / factor) * (size3_input / factor);

	zero_<<<GS(size_output), TB>>>(output, size_output);
	downsample_<<<GS(size_input), TB>>>(input, output, factor, size3_input, size_input);
}

float *x0_gray_big, *x1_gray_big, *x0_gray, *x1_gray, *x0_mc, *x0_disp;
int width_big, height_big, size_big, width, height, size;

int downsample_factor = 4;
int disp_max = 64;

void stereo_init(int width_arg, int height_arg)
{

	width_big = width_arg;
	height_big = height_arg;
	size_big = width_big * height_big;

	assert(width_big % downsample_factor == 0);
	assert(height_big % downsample_factor == 0);

	width = width_big / downsample_factor;
	height = height_big / downsample_factor;
	size = width * height;

	printf("stereo_init: %d x %d\n", width, height);

	ok(cudaMalloc((void **)&x0_gray_big, size_big * 4));
	ok(cudaMalloc((void **)&x1_gray_big, size_big * 4));

	ok(cudaMalloc((void **)&x0_gray, size * 4));
	ok(cudaMalloc((void **)&x1_gray, size * 4));
	ok(cudaMalloc((void **)&x0_mc, size * disp_max * 4));
	ok(cudaMalloc((void **)&x0_disp, size * 4));
}

void stereo_run(unsigned char *x0, unsigned char *x1, unsigned char *display)
{
	rgb2gray_<<<GS(size_big), TB>>>(x0, x0_gray_big, size_big);
	rgb2gray_<<<GS(size_big), TB>>>(x1, x1_gray_big, size_big);

	downsample(x0_gray_big, x0_gray, downsample_factor, height_big, width_big);
	downsample(x1_gray_big, x1_gray, downsample_factor, height_big, width_big);

	ad_<<<GS(size * disp_max), TB>>>(x0_gray, x1_gray, x0_mc, size * disp_max, height, width, -1);
	argmin_<<<GS(size), TB>>>(x0_mc, x0_disp, size, disp_max, size);

	gray2display_<<<GS(size), TB>>>(x0_gray, display, size);
	gray2display_<<<GS(size), TB>>>(x0_disp, display + size * 4, size);
}
