/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
#include <cmath>
#include <vector>
using namespace std;
void correlate(int ny, int nx, const float *data, float *result) {
    vector<double> var_data(ny * nx);
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
        }; double norm = 1 / sqrt(m);
        for (int x = 0; x < nx; x++) {
            var_data[x + y_of] *= norm;
        };
    
    }

    /*
    for (int i = 0; i < ny; i++) {
        int inx = i * nx; int iny = i * ny;
        for (int j = i; j < ny; j++) {
            int jnx = j * nx;
            double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
            int k = 0;
            if (nx >= 4) {
            for (; k <= nx - 4; k += 4) {
                sum0 += var_data[k + jnx] * var_data[k + inx];
                sum1 += var_data[k + 1 + jnx] * var_data[k + 1 + inx];
                sum2 += var_data[k + 2 + jnx] * var_data[k + 2 + inx];
                sum3 += var_data[k + 3 + jnx] * var_data[k + 3 + inx];
                }
            }
            for (;k < nx; k++) {
                sum0 += var_data[k + jnx] * var_data[k + inx];
            }
            result[j + iny] = sum0 + sum1 + sum2 + sum3;
        }
    }*/

    int block_size = 128;
    for (int i_block = 0; i_block < ny; i_block += block_size) {
    for (int j_block = i_block; j_block < ny; j_block += block_size) {
        for (int i = i_block; i < min(i_block + block_size, ny); i++) {
            int inx = i * nx; int iny = i * ny;
            for (int j = j_block; j < min(j_block + block_size, ny); j++) {
            int jnx = j * nx;
            double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
            int k = 0;
            if (nx >= 4) {
            for (; k <= nx - 4; k += 4) {
                sum0 += var_data[k + jnx] * var_data[k + inx];
                sum1 += var_data[k + 1 + jnx] * var_data[k + 1 + inx];
                sum2 += var_data[k + 2 + jnx] * var_data[k + 2 + inx];
                sum3 += var_data[k + 3 + jnx] * var_data[k + 3 + inx];
                }
            }
            for (;k < nx; k++) {
                sum0 += var_data[k + jnx] * var_data[k + inx];
            }
            result[j + iny] = sum0 + sum1 + sum2 + sum3;
            }
        }
    }
}


}