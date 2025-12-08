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
        };
        for (int x = 0; x < nx; x++) {
            var_data[x + y_of] /= sqrt(m);
        };

    }
    for (int j = 0; j < ny; j++) {
        for (int i = j; i < ny; i++) {
            double sum = 0;
            for (int k = 0; k < nx; k++) {
                sum += var_data[k + j * nx] * var_data[k + i * nx];
            }
            result[i + j * ny] = sum;
        }
    }

    for (int j = 0; j < ny; j++) {
        for (int i = j; i < ny; i++) {
            double sum = 0;
            for (int k = 0; k < nx; k++) {
                sum += var_data[k + j * nx] * var_data[k + i * nx];
            }
            result[i + j * ny] = sum;
        }
    }
}


/*
A reasonable way to calculate all pairwise correlations is the following:

First normalize the input rows so that each row has the arithmetic mean of 0 — be careful to do the normalization so that you do not change pairwise correlations.
Then normalize the input rows so that for each row the sum of the squares of the elements is 1 — again, be careful to do the normalization so that you do not change pairwise correlations.
Let X be the normalized input matrix.
Calculate the (upper triangle of the) matrix product Y = XXT.
Now matrix Y contains all pairwise correlations. The only computationally-intensive part is the computation of the matrix product; the normalizations can be done in linear time in the input size.
*/