/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));
constexpr double4_t v0 {0, 0, 0, 0};
static inline double hsum(double4_t vec) {
    return vec[0] + vec[1] + vec[2] + vec[3];
}


#include <cmath>
#include <vector>
using namespace std;
void correlate(int ny, int nx, const float *data, float *result) {
    constexpr int nb = 4; int na = (nx + nb - 1) / nb;
    std::vector<double4_t> vd(ny * na);
    std::vector<double> var_data(ny * nx);
    
    for (int y = 0; y < ny; y++) {
        int y_of = y * nx;
        double m = 0;
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
        for (int x = 0; x < nx; x++) {
            var_data[x + y_of] /= sqrt(m);
        };
    }

    for (int y = 0; y < ny; y++) {
        for (int kx = 0; kx < na; kx++) {
            for (int v = 0; v < nb; v++) {
                int i = kx * nb + v;
                vd[na * y + kx][v] = i < nx ? var_data[nx * y + i] : 0;
            }
        }
    }
    
    int block_size = 32;
    for (int i_block = 0; i_block < ny; i_block += block_size) {
    for (int j_block = i_block; j_block < ny; j_block += block_size) {
        for (int i = i_block; i < min(i_block + block_size, ny); i++) {
            int ina = i * na; int iny = i * ny;
            for (int j = j_block; j < min(j_block + block_size, ny); j++) {
            int jna = j * na;
            double4_t sum0 = v0, sum1 = v0, sum2 = v0, sum3 = v0;
            int k = 0;
            for (; k <= na - 4; k += 4) {
                sum0 += vd[k + jna] * vd[k + ina];
                sum1 += vd[k + 1 + jna] * vd[k + 1 + ina];
                sum2 += vd[k + 2 + jna] * vd[k + 2 + ina];
                sum3 += vd[k + 3 + jna] * vd[k + 3 + ina];
                }
            for (;k < na; k++) {
                sum0 += vd[k + jna] * vd[k + ina];
            }
            result[j + iny] = hsum(sum0) + hsum(sum1) + hsum(sum2) + hsum(sum3);
            }
        }
    }
}
}