#include <vector>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

#define CHECK(x) check(x, #x)

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

__global__ void mtmmult(float *result, const float *data, int ny, int nx) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= ny || j >= ny) return;
    
    float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    int z = 0;
    for (; z < nx - 3; z += 4) {
        sum0 += data[z + nx * i] * data[z + nx * j];
        sum1 += data[z + 1 + nx * i] * data[z + 1 + nx * j];
        sum2 += data[z + 2 + nx * i] * data[z + 2 + nx * j];
        sum3 += data[z + 3 + nx * i] * data[z + 3 + nx * j];
    }
    for (; z < nx; ++z) {
        sum0 += data[z + nx * i] * data[z + nx * j];
    }
    float s = sum0 + sum1 + sum2 + sum3;
    result[i + ny * j] = s;
}

void correlate(int ny, int nx, const float *data, float *result) {
    
    int grid_y = divup(ny, 16), grid_x = divup(ny, 16);
    std::vector<float> var_data(ny * nx, 0.0f);

    //preprocess data
    for (int y = 0; y < ny; y++) {
        int y_of = y * nx;
        float m = 0;
        for (int x = 0; x < nx; x++) {
            m += data[x + y_of];
        }; if (nx) {m /= nx;}
        for (int x = 0; x < nx; x++) {
            var_data[x + y_of] = data[x + y_of] - m;
        };
        m = 0;
        for (int x = 0; x < nx; x++) {
            m += var_data[x + y_of] * var_data[x + y_of];
        };
        m = sqrtf(m);
        if (m < 1e-6f) m = 1.0f;
        for (int x = 0; x < nx; x++) {
            var_data[x + y_of] /= m;
        };
    }

    // allocate memory & copy data to GPU
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, ny * nx * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, var_data.data(), ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    // Run algorithm
    {
        // these 2 will directly define what blocks will be processed
        dim3 dimGrid(grid_x, grid_y); // rounds dimensions up such that all data is covered, x and y define grid x and y
        dim3 dimBlock(16, 16); // standard block size definition
        mtmmult<<<dimGrid, dimBlock>>>(rGPU, dGPU, ny, nx);
        CHECK(cudaGetLastError());
    }
    cudaDeviceSynchronize();

    // Copy data back to CPU (straight to result) and free data
    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));

}
