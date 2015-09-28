#include <stdio.h>
#include <assert.h>
#include <math_constants.h>
#include <sys/time.h>

#include <cudnn.h>

#define TB 128
#define GS(x) (((x) - 1) / TB + 1)

#define ok(expr) if (expr != 0) { printf("ERROR on line %d\n", __LINE__); exit(-1); }

/* Tensor */
struct Tensor {
    float *data;
    cudnnTensorDescriptor_t desc;
    int n, c, h, w, size, capacity;
};

void Tensor_init(struct Tensor *t)
{
    t->data = NULL;
    t->n = t->c = t->h = t->w = t->size = t->capacity = 0;
    ok(cudnnCreateTensorDescriptor(&t->desc));
}

void Tensor_resize(struct Tensor *t, int n, int c, int h, int w)
{
    int size = n * c * h * w;

    if (t->capacity != 0 && size > t->capacity) {
        printf("DNN: reallocating tensor\n");
        ok(cudaFree(t->data));
        t->capacity = 0;
    }

    if (t->capacity == 0) {
        t->capacity = size;
        ok(cudaMalloc(&t->data, t->capacity * 4));
    }
    t->n = n;
    t->c = c;
    t->h = h;
    t->w = w;
    t->size = size;

    ok(cudnnSetTensor4dDescriptor(t->desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w))
}

double get_time()
{
    struct timeval t;
    struct timezone tzp;
    gettimeofday(&t, &tzp);
    return t.tv_sec + t.tv_usec * 1e-6;
}

__global__ void zero_(float *input, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
		input[id] = 0;
	}
}

void zero(Tensor *t)
{
	zero_<<<GS(t->size), TB>>>(t->data, t->size);
}

__global__ void mul_(float *input, float factor, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
		input[id] = input[id] * factor;
	}
}

void mul(Tensor *t, float factor)
{
	mul_<<<GS(t->size), TB>>>(t->data, factor, t->size);
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

void rgb2gray(unsigned char *input, Tensor *output, int h, int w)
{
	Tensor_resize(output, 1, 1, h, w);
	rgb2gray_<<<GS(output->size), TB>>>(input, output->data, output->size);
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

void gray2display(Tensor *input, unsigned char *display)
{
	gray2display_<<<GS(input->size), TB>>>(input->data, display, input->size);
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

void ad(Tensor *x0, Tensor *x1, Tensor *mc, int disp_max, int direction)
{
	Tensor_resize(mc, 1, disp_max, x0->h, x0->w);
	ad_<<<GS(mc->size), TB>>>(x0->data, x1->data, mc->data, mc->size, mc->h, mc->w, direction);
}


__global__ void argmin_(float *input, float *output, int size1, int size23)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size23) {
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

void argmin(Tensor *input, Tensor *output)
{
	Tensor_resize(output, 1, 1, input->h, input->w);
	argmin_<<<GS(output->size), TB>>>(input->data, output->data, input->c, output->size);
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

void downsample(Tensor *input, Tensor *output, int factor) //, int size2_input, int size3_input)
{
	assert(input->h % factor == 0);
	assert(input->w % factor == 0);

	Tensor_resize(output, 1, 1, input->h / factor, input->w / factor);

	zero(output);
	downsample_<<<GS(input->size), TB>>>(input->data, output->data, factor, input->w, input->size);
}

cudnnHandle_t cudnn_handle;

Tensor x0_gray_big, x1_gray_big, x0_gray, x1_gray, x0_mc, x0_disp;
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

	ok(cudnnCreate(&cudnn_handle));
	
	Tensor_init(&x0_gray_big);
	Tensor_init(&x1_gray_big);
	Tensor_init(&x0_gray);
	Tensor_init(&x1_gray);
	Tensor_init(&x0_mc);
	Tensor_init(&x0_disp);
}

void stereo_run(unsigned char *x0, unsigned char *x1, unsigned char *display)
{
	rgb2gray(x0, &x0_gray_big, height_big, width_big);
	rgb2gray(x1, &x1_gray_big, height_big, width_big);

	downsample(&x0_gray_big, &x0_gray, downsample_factor);
	downsample(&x1_gray_big, &x1_gray, downsample_factor);

	ad(&x0_gray, &x1_gray, &x0_mc, disp_max, -1);
	argmin(&x0_mc, &x0_disp);

	gray2display(&x0_gray, display);
	gray2display(&x0_disp, display + size * 4);
}
