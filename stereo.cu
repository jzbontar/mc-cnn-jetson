#include <assert.h>
#include <fcntl.h>
#include <math_constants.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>

#include <cudnn.h>

#define TB 128
#define GS(x) (((x) - 1) / TB + 1)

#define ok(expr) if (expr != 0) { printf("ERROR on line %d\n", __LINE__); exit(-1); }

cudnnHandle_t cudnn_handle;

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

void Tensor_init_resize(struct Tensor *t, int n, int c, int h, int w)
{
	Tensor_init(t);
	Tensor_resize(t, n, c, h, w);
}

void Tensor_print_ext(struct Tensor *t, int n, int c, int h, int w)
{
	float *buf = (float *)malloc(t->size * 4);
	cudaMemcpy(buf, t->data, t->size * 4, cudaMemcpyDeviceToHost);
	printf("%d x %d x %d x %d; %d, %d, %d, %d\n", t->n, t->c, t->h, t->w, n, c, h, w);
	assert(n < t->n);
	assert(c < t->c);
	int i = n;
	int j = c;
	for (int k = h; k < min(t->h, h + 6); k++) {
		for (int l = w; l < min(t->w, w + 6); l++) {
			printf("%e ", buf[((i * t->c + j) * t->h + k) * t->w + l]);
		}
		printf("\n");
	}
	free(buf);
}

void Tensor_print(struct Tensor *t)
{
	Tensor_print_ext(t, 0, 0, 0, 0);
}

__global__ void fill_(float *input, float value, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		input[id] = value;
	}
}

void fill(Tensor *t, float value)
{
	fill_<<<GS(t->size), TB>>>(t->data, value, t->size);
}

__global__ void add_(float *input, float value, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		input[id] = input[id] + value;
	}
}

void add(Tensor *t, float value)
{
	add_<<<GS(t->size), TB>>>(t->data, value, t->size);
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

/* ConvLayer */
struct ConvLayer {
	int relu;
	struct Tensor output, weight, bias;
	cudnnConvolutionFwdAlgo_t algorithm;
	cudnnFilterDescriptor_t weight_desc;
	cudnnConvolutionDescriptor_t conv_desc;
};

void ConvLayer_init(ConvLayer *e, int n_in, int n_out, int kw, int kh, int sx, int sy, int padw, int padh, int relu)
{
	Tensor_init(&e->output);
	Tensor_init_resize(&e->weight, n_out, n_in, kh, kw);
	Tensor_init_resize(&e->bias, 1, n_out, 1, 1);
	ok(cudnnCreateFilterDescriptor(&e->weight_desc));
	ok(cudnnSetFilter4dDescriptor(e->weight_desc, CUDNN_DATA_FLOAT, n_out, n_in, kh, kw));
	ok(cudnnCreateConvolutionDescriptor(&e->conv_desc));
	ok(cudnnSetConvolution2dDescriptor(e->conv_desc, padh, padw, sy, sx, 1, 1, CUDNN_CROSS_CORRELATION));
	e->relu = relu;
}

struct Tensor *ConvLayer_allocate(ConvLayer *e, struct Tensor *i) {
	int n, c, h, w;
	ok(cudnnGetConvolution2dForwardOutputDim(e->conv_desc, i->desc, e->weight_desc, &n, &c, &h, &w));
	Tensor_resize(&e->output, n, c, h, w);
	fill(&e->output, 0);
	ok(cudnnGetConvolutionForwardAlgorithm(cudnn_handle, i->desc, e->weight_desc, e->conv_desc,
		e->output.desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &e->algorithm));
	return &e->output;
}

struct Tensor *ConvLayer_forward(ConvLayer *e, struct Tensor *i) {
	float zero = 0;
	float one = 1;
	ok(cudnnConvolutionForward(cudnn_handle, &one, i->desc, i->data, e->weight_desc, e->weight.data,
		e->conv_desc, e->algorithm, NULL, 0, &zero, e->output.desc, e->output.data));
	ok(cudnnAddTensor(cudnn_handle, CUDNN_ADD_SAME_C, &one, e->bias.desc, e->bias.data, &one,
		e->output.desc, e->output.data));
	if (e->relu) {
		ok(cudnnActivationForward(cudnn_handle, CUDNN_ACTIVATION_RELU, &one, e->output.desc, 
			e->output.data, &zero, e->output.desc, e->output.data));
	}
	return &e->output;
}

/* Sequential */
struct Sequential {
	struct ConvLayer modules[32];
	int num_modules;
};

void mmap2gpu(const char *fname, float *data, int size)
{
	int fd = open(fname, O_RDONLY);
	float *map = (float *)mmap(NULL, size * 4, PROT_READ, MAP_SHARED, fd, 0);
	cudaMemcpy(data, map, size, cudaMemcpyHostToDevice);
	close(fd);
}

void Sequential_load(struct Sequential *s, const char *dir)
{
	char buf[256];
	snprintf(buf, 256, "%s/desc", dir);
	FILE *f = fopen(buf, "r");
	int n = fscanf(f, "%d\n", &s->num_modules);

	printf("load network from %s with %d conv layers\n", dir, s->num_modules);
	for (int i = 0; i < s->num_modules; i++) {
		int n_in, n_out, kw, kh, dw, dh, padw, padh, relu;
		n = fscanf(f, "%d %d %d %d %d %d %d %d %d\n", &n_in, &n_out, &kw, &kh, &dw, &dh, &padw, &padh, &relu);
		ConvLayer_init(s->modules + i, n_in, n_out, kw, kh, dw, dh, padw, padh, relu);
		printf("conv: %d %d %d %d %d %d %d %d %d\n", n_in, n_out, kw, kh, dw, dh, padw, padh, relu);

		snprintf(buf, 256, "%s/%dW", dir, i);
		mmap2gpu(buf, s->modules[i].weight.data, s->modules[i].weight.size * 4);
		snprintf(buf, 256, "%s/%dB", dir, i);
		mmap2gpu(buf, s->modules[i].bias.data, s->modules[i].bias.size * 4);
	}
}

struct Tensor *Sequential_allocate(struct Sequential *s, struct Tensor *input)
{
	struct Tensor *output = input;
	for (int i = 0; i < s->num_modules; i++) {
		output = ConvLayer_allocate(s->modules + i, output);
	}
	return output;
}

struct Tensor *Sequential_forward(struct Sequential *s, struct Tensor *input)
{
	struct Tensor *output = input;
	for (int i = 0; i < s->num_modules; i++) {
		output = ConvLayer_forward(s->modules + i, output);
	}
	return output;
}

double get_time()
{
	struct timeval t;
	struct timezone tzp;
	gettimeofday(&t, &tzp);
	return t.tv_sec + t.tv_usec * 1e-6;
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

__global__ void Normalize_get_norm_(float *input, float *norm, int size1, int size23, int size023)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size023) {
		int dim23 = id % size23;
		int dim0 = id / size23;

		float sum = 0.0;
		for (int dim1 = 0; dim1 < size1; dim1++) {
			float x = input[(dim0 * size1 + dim1) * size23 + dim23];
			sum += x * x;
		}
		norm[dim0 * size23 + dim23] = sum + 1.3e-37;
	}
}

__global__ void Normalize_forward_(float *input, float *norm, float *output, int size23, int size123, int size0123)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size0123) {
		int dim23 = id % size23;
		int dim0 = (id / size123);
		output[id] = input[id] / sqrtf(norm[dim0 * size23 + dim23]);
	}
}

void Normalize_forward(Tensor *input, Tensor *norm)
{
	Tensor_resize(norm, input->n, 1, input->h, input->w);
	Normalize_get_norm_<<<GS(norm->size), TB>>>(input->data, norm->data, input->c, 
		input->h * input->w, norm->size);
	Normalize_forward_<<<GS(input->size), TB>>>(input->data, norm->data, input->data,
		input->h * input->w, input->c * input->h * input->w, input->size);
}

__global__ void StereoJoin_(float *input_L, float *input_R, float *output_L, float *output_R, int size1_input, int size1, int size3, int size23)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size23) {
		int dim3 = id % size3;
		assert(size1_input <= 32);
		float L_cache[32];
		for (int i = 0; i < size1_input; i++) {
			L_cache[i] = input_L[i * size23 + id];
		}

		for (int d = 0; d < size1; d++) {
			if (dim3 - d >= 0) {
				float sum = 0;
				for (int i = 0; i < size1_input; i++) {
					sum -= L_cache[i] * input_R[i * size23 + id - d];
				}
				output_L[d * size23 + id] = sum;
				output_R[d * size23 + id - d] = sum;
			}
		}
	}
}

void StereoJoin(Tensor *input, Tensor *output_L, Tensor *output_R, int disp_max)
{
	Tensor_resize(output_L, 1, disp_max, input->h, input->w);
	Tensor_resize(output_R, 1, disp_max, input->h, input->w);
	fill(output_L, NAN);
	fill(output_R, NAN);
	int size23 = input->h * input->w;
	StereoJoin_<<<GS(size23), TB>>>(input->data, input->data + input->size / 2, output_L->data, output_R->data,
		input->c, output_L->c, output_L->w, size23);
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
			dist = CUDART_NAN_F;
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
		float min = CUDART_INF_F;
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

void downsample(Tensor *input, Tensor *output, int factor)
{
	assert(input->h % factor == 0);
	assert(input->w % factor == 0);
	Tensor_resize(output, 1, 1, input->h / factor, input->w / factor);
	fill(output, 0);
	downsample_<<<GS(input->size), TB>>>(input->data, output->data, factor, input->w, input->size);
}

void load_batch(Tensor *x0, Tensor *x1, Tensor *batch)
{
	Tensor_resize(batch, 2, 1, x0->h, x0->w);
	cudaMemcpy(batch->data, x0->data, x0->size * 4, cudaMemcpyDeviceToDevice);
	cudaMemcpy(batch->data + x0->size, x1->data, x0->size * 4, cudaMemcpyDeviceToDevice);
}

Sequential net;
Tensor x0_gray_big, x1_gray_big, x0_gray, x1_gray, x0_mc, x1_mc, x0_disp, x1_disp, batch, norm;
int width_big, height_big, size_big, width, height, size;

int downsample_factor = 10;
int disp_max = 32;
const float mean = 95;
const float stddev = 65;

void stereo_init(int width_arg, int height_arg)
{
	Tensor_init(&x0_gray_big);
	Tensor_init(&x1_gray_big);
	Tensor_init(&x0_gray);
	Tensor_init(&x1_gray);
	Tensor_init(&x0_mc);
	Tensor_init(&x1_mc);
	Tensor_init(&x0_disp);
	Tensor_init(&x1_disp);
	Tensor_init(&batch);
	Tensor_init(&norm);

	width_big = width_arg;
	height_big = height_arg;
	size_big = width_big * height_big;

	assert(width_big % downsample_factor == 0);
	assert(height_big % downsample_factor == 0);

	width = width_big / downsample_factor;
	height = height_big / downsample_factor;
	size = width * height;

	ok(cudnnCreate(&cudnn_handle));

	Tensor_resize(&batch, 2, 1, height, width);
	Sequential_load(&net, "net/net_kitti_fast_-a_train_tr_-fm_16");
	Sequential_allocate(&net, &batch);

	printf("stereo_init: %d x %d\n", width, height);
}

void stereo_run(unsigned char *x0, unsigned char *x1, unsigned char *display)
{
	rgb2gray(x0, &x0_gray_big, height_big, width_big);
	rgb2gray(x1, &x1_gray_big, height_big, width_big);

	downsample(&x0_gray_big, &x0_gray, downsample_factor);
	downsample(&x1_gray_big, &x1_gray, downsample_factor);

	// image preprocessing
	add(&x0_gray, -mean);
	mul(&x0_gray, 1 / stddev);
	add(&x1_gray, -mean);
	mul(&x1_gray, 1 / stddev);

	// network
	load_batch(&x0_gray, &x1_gray, &batch);
	Tensor *output = Sequential_forward(&net, &batch);
	Normalize_forward(output, &norm);
	StereoJoin(output, &x0_mc, &x1_mc, disp_max);

	// stereo method
	argmin(&x0_mc, &x0_disp);
	mul(&x0_disp, 8);

	// absolute differences
	// ad(&x0_gray, &x1_gray, &x0_mc, disp_max, -1);
	// argmin(&x0_mc, &x0_disp);

	// undo image preprocessing
	mul(&x0_gray, stddev);
	add(&x0_gray, mean);
	mul(&x1_gray, stddev);
	add(&x1_gray, mean);

	gray2display(&x0_gray, display);
	gray2display(&x0_disp, display + size * 4);
}
