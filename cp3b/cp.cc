typedef float float8_t __attribute__ ((vector_size (8 * sizeof(float))));
constexpr float8_t v0 {0, 0, 0, 0, 0, 0, 0, 0};
static inline float hsum(float8_t vec) {
    return vec[0] + vec[1] + vec[2] + vec[3] + vec[4] + vec[5] + vec[6] + vec[7];
}


#include <cmath>
#include <vector>
#include <immintrin.h>


using namespace std;
void correlate(int ny, int nx, const float *data, float *result) {
    constexpr int nb = 8; int na = (nx + nb - 1) / nb;
    alignas(64) std::vector<float8_t> vd(ny * na);
    std::vector<float> var_data(ny * nx);
    alignas(64) std::vector<float8_t> result_v(ny * ny, v0);

    
    
    #pragma omp parallel for
    for (int y = 0; y < ny; y++) {
        int y_of = y * nx;
        double m = 0;
        #pragma omp simd
        for (int x = 0; x < nx; x++) {
            m += data[x + y_of];
        }; if (nx) {m /= nx;}
        #pragma omp simd
        for (int x = 0; x < nx; x++) {
            var_data[x + y_of] = data[x + y_of] - m;
        };
        m = 0;
        #pragma omp simd
        for (int x = 0; x < nx; x++) {
            m += var_data[x + y_of] * var_data[x + y_of];
        };
        #pragma omp simd
        for (int x = 0; x < nx; x++) {
            var_data[x + y_of] /= sqrt(m);
        };
    }


    #pragma omp parallel for
    for (int y = 0; y < ny; y++) {
        for (int kx = 0; kx < na; kx++) {
            #pragma omp simd
            for (int v = 0; v < nb; v++) {
                int i = kx * nb + v;
                //vd[na * y + kx][v] = i < nx ? var_data[nx * y + i] : 0;
                vd[ny * kx + y][v] = i < nx ? var_data[nx * y + i] : 0;
            }
        }
    }


    constexpr int size_i = 125, size_j = 125, size_k = 50;
    #pragma omp parallel for schedule(dynamic)
    for (int i_b = 0; i_b < ny; i_b += size_i) {
    for (int k_b = 0; k_b < na; k_b += size_k) {
    for (int j_b = i_b; j_b < ny; j_b += size_j) {
        for (int i = i_b; i < min(i_b + size_i, ny); i++) {
        for (int k = k_b; k < min(k_b + size_k, na); k++) {
        float8_t i_val = vd[k * ny + i]; 
        #pragma omp simd
        for (int j = j_b; j < min(j_b + size_j, ny); j++) {
            result_v[i * ny + j] += i_val * vd[k * ny + j];
        }
        }    
        }
    }
    }
    }

    #pragma omp distribute parallel for simd
    for (int i = 0; i < ny * ny; i++) {
        result[i] = hsum(result_v[i]);
    }
    
}
