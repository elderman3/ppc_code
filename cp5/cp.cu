#include <vector>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

#define CHECK(x) check(x, #x)
#define KIT 16
#define XS 2
#define YS 32
#define KS 16
#define TILE_DIM 32
#define BLOCK_ROWS 8



static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

__global__ void mtmmult(float* __restrict__ result, const float* __restrict__ data, int ny, int nx) {
    int i = KS * (threadIdx.x + blockIdx.x * blockDim.x);
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    __shared__ float dt[KIT][YS + 1];
    float sum[KS];
    int max = min(KS, ny - i);
    int e = min(i+KS-1, ny - 1);
    for (int b = 0; b < KS; ++b) sum[b] = 0.f;
    int tx = threadIdx.x, ty = threadIdx.y;

    for (int k = 0; k < nx; k += KIT) {
    int w = min(KIT, nx - k);

    int piter = KIT / XS;
    int t = tx * piter;
    #pragma unroll
    for (int p = 0; p < piter; ++p) {
        int dp = j + (t + k + p) * ny;
        dt[t + p][ty] = (dp < ny * nx) ? data[dp] : 0.f;
    }
    __syncthreads();

    
    if (j < ny && e >= j) {
        for (int s = 0; s < w; ++s) {
            float v = dt[s][ty];
            int bs = (k + s) * ny + i;
            #pragma unroll
            for (int d = 0; d < max; ++d) {
                sum[d] += v * data[bs + d];
            }
        }
    }
    __syncthreads();
    }
    if (j < ny && e >= j) {
        #pragma unroll
        for (int s = 0; s < max; ++s) {
            result[i + s + j * ny] = sum[s];
        }
        }
}

__global__ void prepr(float* __restrict__ result, const float* __restrict__ data, int ny, int nx) {
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= ny) return;
    int ynx = y * nx;
    float sum = 0.f;
    for (int x = 0; x < nx; x++) {
        sum += data[x + ynx];
    } if (nx) {sum /= nx;}
    for (int x = 0; x < nx; ++x) {
        result[x + ynx] = data[x + ynx] - sum;
    }
    float m = 0.f;
    for (int x = 0; x < nx; ++x) {
        m += result[x + ynx] * result[x + ynx];
    }
    m = rsqrtf(m);
    if (m < 1e-6f) m = 1.0f;
    for (int x = 0; x < nx; x++) {
        result[x + ynx] *= m;
    }
}

__global__ void transpose(float *odata, float *idata, int nx, int ny) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; 

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Read the tile from global memory
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if ((y + i) < ny && x < nx)
            tile[threadIdx.y + i][threadIdx.x] = idata[(y + i) * nx + x];
    }

    __syncthreads();

    // Write the transposed tile to global memory
    x = blockIdx.y * TILE_DIM + threadIdx.x; // swap blockIdx.x and blockIdx.y
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if ((y + i) < nx && x < ny)
            odata[(y + i) * ny + x] = tile[threadIdx.x][threadIdx.y + i];
    }
}

void correlate(int ny, int nx, const float *data, float *result) {
    printf("x%i, y%i, k%i, it%i", XS, YS, KS, KIT);
    int grid_x = divup(ny, XS * KS), grid_y = divup(ny, YS);
    std::vector<float> vd(ny * nx, 0.0f);


    float* dpGPU = NULL;
    CHECK(cudaMalloc((void**)&dpGPU, ny * nx * sizeof(float)));
    float* rpGPU = NULL;
    CHECK(cudaMalloc((void**)&rpGPU, ny * nx * sizeof(float)));
    CHECK(cudaMemcpy(dpGPU, data, ny * nx * sizeof(float), cudaMemcpyHostToDevice));
    {
        dim3 dimGrid(1, divup(ny, 32));
        dim3 dimBlock(1, 32);
        prepr<<<dimGrid, dimBlock>>>(rpGPU, dpGPU, ny, nx);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaFree(dpGPU));

    float* rtGPU = NULL;
    CHECK(cudaMalloc((void**)&rtGPU, ny * nx * sizeof(float)));

        dim3 grid((nx + TILE_DIM - 1) / TILE_DIM, (ny + TILE_DIM - 1) / TILE_DIM);
        dim3 block(TILE_DIM, BLOCK_ROWS);
        transpose<<<grid, block>>>(rtGPU, rpGPU, nx, ny);
        CHECK(cudaGetLastError());

    CHECK(cudaFree(rpGPU));
    
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemset(rGPU, 0, ny * ny * sizeof(float)));
    // Run algorithm
    {
        dim3 dimGrid(grid_x, grid_y);
        dim3 dimBlock(XS, YS); a
        mtmmult<<<dimGrid, dimBlock>>>(rGPU, rtGPU, ny, nx);
        CHECK(cudaGetLastError());
    }
    //cudaDeviceSynchronize();
    // Copy data back to CPU (straight to result) and free data
    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(rGPU));
    CHECK(cudaFree(rtGPU));
}