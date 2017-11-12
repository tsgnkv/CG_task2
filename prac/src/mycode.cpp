#include "../include/mycode.h"


/**
@file mycode.cpp
*/

/// size of rescaled image
#define IMAGE_SIZE 64
/// width of cell
#define WIDTH 16
/// height of cell
#define HEIGHT 8
/// segment patrition number
#define SEGMENT 16
/// pi
#define PI 3.1415962

/**
@function grayscale
convert RGBA image to grayscale
@param in_input is an input image.
@param out_mat is an output matrix of double.
*/
void grayscale(BMP *image, Matrix<float> &matrix) {

    Rescale(*image , 'F', IMAGE_SIZE);
    
    int height = image->TellHeight();
    int width = image->TellWidth();

    if (height > width) {
        int border = (height - width) / 2;
        // initialize borders
        for (int i = 0; i < IMAGE_SIZE; i++) {
            for(int j = 0; j <= border; j++) {
                matrix(i, j) = 0;
                matrix(IMAGE_SIZE - 1 - i, j) = 1;
            }
        }
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                matrix(j, i + border) = 0.299*((*image)(i, j))->Red + 0.587*((*image)(i, j))->Green + 0.114*((*image)(i, j))->Blue;
            }
        }
    } else {
        int border = (width - height) / 2;
        // initialize borders
        for (int i = 0; i <= border; i++) {
            for(int j = 0; j < IMAGE_SIZE; j++) {
                matrix(i, j) = 0;
                matrix(IMAGE_SIZE - 1 - i, j) = 1;
            }
        }
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                matrix(j + border, i) = 0.299*((*image)(i, j))->Red + 0.587*((*image)(i, j))->Green + 0.114*((*image)(i, j))->Blue;
            }
        }
    }
}

/**
@function sobel
apply sobel filter to grayscale matrix
@param src is an input grayscale matrix.
@param sobel_x is an output sobel_x matrix.
@param sobel_y is an output sobel_y matrix.
*/
void sobel(const Matrix<float> &src, Matrix<float> &sobel_x, Matrix<float> &sobel_y) {

    static const Matrix<float> kernel_x = { {-1, 0, 1},
                                            {-2, 0, 2},
                                            {-1, 0, 1}};

    static const Matrix<float> kernel_y = { {-1,-2,-1},
                                            { 0, 0, 0},
                                            { 1, 2, 1}};

    auto tmp = src.extra_borders(1, 1);

    double sum_x, sum_y;

    for (int k = 0; k < IMAGE_SIZE; k++) {
        for (int m = 0; m < IMAGE_SIZE; m++) {

            sum_x = 0;
            sum_y = 0;

            for (uint i = 0; i < kernel_x.n_rows; i++) {
                for (uint j = 0; j < kernel_x.n_cols; j++) {
                    sum_x += tmp(k + i, m + j) * kernel_x(i, j);
                    sum_y += tmp(k + i, m + j) * kernel_y(i, j);
                }
            }

            sobel_x(k, m) = sum_x;
            sobel_y(k, m) = sum_y;
        }
    }
}

/**
@function sobelSSE
apply sobel filter to grayscale matrix, but it works with SSE
@param src is an input grayscale matrix.
@param sobel_x is an output sobel_x matrix.
@param sobel_y is an output sobel_y matrix.
*/
void sobelSSE(const Matrix<float> &src, Matrix<float> &sobel_x, Matrix<float> &sobel_y) {
    static const __m128 first_x =  _mm_set_ps(0.0, 1.0, 0.0,-1.0);
    static const __m128 second_x = _mm_set_ps(0.0, 2.0, 0.0,-2.0);
    static const __m128 third_x =  _mm_set_ps(0.0, 1.0, 0.0,-1.0);

    static const __m128 first_y =  _mm_set_ps(0.0,-1.0,-2.0,-1.0);
    static const __m128 third_y =  _mm_set_ps(0.0, 1.0, 2.0, 1.0);

    __m128 x_first;
    __m128 x_second;
    __m128 x_third;

    __m128 y_first;
    __m128 y_third;

    auto tmp = src.extra_borders(1, 1);

    for (int k = 1; k < IMAGE_SIZE; k++) {
        for (int m = 1; m < IMAGE_SIZE; m++) {

            x_first = _mm_set_ps(0.0, tmp(k - 1, m + 1), tmp(k - 1, m), tmp(k - 1, m - 1));
            x_second = _mm_set_ps(0.0, tmp(k, m + 1), tmp(k, m), tmp(k, m - 1));
            x_third = _mm_set_ps(0.0, tmp(k + 1, m + 1), tmp(k + 1, m), tmp(k + 1, m - 1));

            y_first = _mm_set_ps(0.0, tmp(k - 1, m + 1), tmp(k - 1, m), tmp(k - 1, m - 1));
            y_third = _mm_set_ps(0.0, tmp(k + 1, m + 1), tmp(k + 1, m), tmp(k + 1, m - 1));

            x_first = _mm_mul_ps(x_first, first_x);
            x_second = _mm_mul_ps(x_second, second_x);
            x_third = _mm_mul_ps(x_third, third_x);

            y_first = _mm_mul_ps(y_first, first_y);
            y_third = _mm_mul_ps(y_third, third_y);

            sobel_x(k, m) = x_first[0] + x_first[2] +
                            x_second[0] + x_second[2] +
                            x_third[0] + x_third[2];
            sobel_y(k, m) = y_first[0] + y_first[1] + y_first[2] +
                            y_third[0] + y_third[1] + y_third[2];
        }
    }
}

/**
@function abs_and_angles
find gradient absolute value and its angle
@param sobel_x is input matrix with x coordinate.
@param sobel_y is input matrix with y coordinate.
@param abs is output matrix with absolute value.
@param angles is output matrix with angles.
*/
void abs_and_angles(const Matrix<float> &sobel_x, const Matrix<float> &sobel_y, Matrix<float> &abs, Matrix<float> &angles) {
    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            abs(i, j) = sqrt(sobel_x(i, j) * sobel_x(i, j) + sobel_y(i, j) * sobel_y(i, j));
            angles(i, j) = std::atan2(sobel_y(i,j), sobel_x(i, j)) + PI;
        }
    }
}

/**
@function abs_and_angles
find gradient absolute value and its angle, but calculating absolute value performed with SSE
@param sobel_x is input matrix with x coordinate.
@param sobel_y is input matrix with y coordinate.
@param abs is output matrix with absolute value.
@param angles is output matrix with angles.
*/
void abs_and_anglesSSE(Matrix<float> &sobel_x, Matrix<float> &sobel_y, Matrix<float> &abs, Matrix<float> &angles) {
    float *data_x = sobel_x.get_ptr();
    float *data_y = sobel_y.get_ptr();

    __m128 x;
    __m128 y;
    __m128 sum;

    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < IMAGE_SIZE / 4; j++) {

            x = _mm_loadu_ps(data_x + i * IMAGE_SIZE + j * 4);
            y = _mm_loadu_ps(data_y + i * IMAGE_SIZE + j * 4);

            x = _mm_mul_ps(x, x);
            y = _mm_mul_ps(y, y);

            sum = _mm_add_ps(x, y);
            sum = _mm_sqrt_ps(sum);

            abs(i, j * 4) = sum[0];
            abs(i, j * 4 + 1) = sum[1];
            abs(i, j * 4 + 2) = sum[2];
            abs(i, j * 4 + 3) = sum[3];

            angles(i, j * 4) = std::atan2(sobel_y(i,j * 4), sobel_x(i, j * 4)) + PI;
            angles(i, j * 4 + 1) = std::atan2(sobel_y(i,j * 4 + 1), sobel_x(i, j * 4 + 1)) + PI;
            angles(i, j * 4 + 2) = std::atan2(sobel_y(i,j * 4 + 2), sobel_x(i, j * 4 + 2)) + PI;
            angles(i, j * 4 + 3) = std::atan2(sobel_y(i,j * 4 + 3), sobel_x(i, j * 4 + 3)) + PI;

        }
    }
}

/**
@function calc_histogram
calculate histogram for cells of matrix and adding features to vector
@param abs is input matrix with absolute values.
@param angles is input matrix with angles.
@param features is output vector for features.
@param h is height of cell.
@param w is width of cell.
*/
void calc_histogram(const Matrix<float> &abs, const Matrix<float> &angels, vector<float> &features, int h, int w) {

    int vertical = IMAGE_SIZE / h;
    int horizontal = IMAGE_SIZE / w;

    for (int i = 0; i < vertical; i++) {
        for (int j = 0; j < horizontal; j++) {

            vector<float> histogram(SEGMENT, 0);

            for (int k = 0; k < h; k++) {
                for (int m = 0; m < w; m++) {
                    int id = static_cast<uint>(angels(i * h + k, j * w + m) * SEGMENT / (2.0 * PI));
                    histogram[id] += abs(i * h + k, j * w + m);
                }
            }

            double norm = 0;
            for (int l = 0; l < SEGMENT; l++) {
                norm += histogram[l] * histogram[l];
            }

            if (norm > 0) {
                norm = sqrt(norm);
                for (int l = 0; l < SEGMENT; l++) {
                    histogram[l] /= norm;
                }
            }

            features.insert(features.end(), histogram.begin(), histogram.end());

        }
    }
}

/**
@function HOG
extract HOG features from image
@param features is output vector for features.
@param gray is grayscale matrix of image.
@param sobel_x is matrix for x coordinate sobel filter.
@param sobel_y is matrix for y coordinate sobel filter.
@param abs is matrix for gradient absolute values.
@param angles is matrix for gradien angles.
*/
void HOG(vector<float> &features, Matrix<float> &gray, Matrix<float> &sobel_x, Matrix<float> &sobel_y, Matrix<float> &abs, Matrix<float> &angels) {
    //sobel(gray, sobel_x, sobel_y);
    sobelSSE(gray, sobel_x, sobel_y);
    //abs_and_angles(sobel_x, sobel_y, abs, angels);
    abs_and_anglesSSE(sobel_x, sobel_y, abs, angels);
    calc_histogram(abs, angels, features, HEIGHT, WIDTH);
}

/**
@function LBP
extract LBP features from image and add its to vector
@param gray is grayscale matrix of image.
@param features is output vector for features.
*/
void LBP(Matrix<float> &gray, vector<float> &features) {

    gray = gray.extra_borders(1, 1);

    int vertical = IMAGE_SIZE / HEIGHT;
    int horizontal = IMAGE_SIZE / WIDTH;
    int id;
    float center;

    for (int i = 0; i < vertical; i++) {
        for (int j = 0; j < horizontal; j++) {

            vector<float> histogram(256, 0);

            for (int k = 0; k < HEIGHT; k++) {
                for (int m = 0; m < WIDTH; m++) {

                    center = gray(vertical * i + 1 + k, horizontal * i + 1 + m);
                    
                    id = 0;

                    id |= (gray(vertical * i + k, horizontal * i + m) > center) << 7;
                    id |= (gray(vertical * i + k, horizontal * i + m + 1) > center) << 6;
                    id |= (gray(vertical * i + k, horizontal * i + m + 2) > center) << 5;
                    id |= (gray(vertical * i + k + 1, horizontal * i + m + 2) > center) << 4;
                    id |= (gray(vertical * i + k + 2, horizontal * i + m + 2) > center) << 3;
                    id |= (gray(vertical * i + k + 2, horizontal * i + m + 1) > center) << 2;
                    id |= (gray(vertical * i + k + 2, horizontal * i + m) > center) << 1;
                    id |= (gray(vertical * i + k + 1, horizontal * i + m) > center) << 0;

                    histogram[id]++;
                }
            }

            double norm = 0;
            for (int l = 0; l < 256; l++) {
                norm += histogram[l] * histogram[l];
            }

            norm = sqrt(norm);

            if (norm > 0) {
                for (int l = 0; l < 256; l++) {
                    histogram[l] /= norm;
                }
            }

            features.insert(features.end(), histogram.begin(), histogram.end());

        }
    }
}

/**
@function color
extract color features from image
@param image is input image.
@param features is output vector for features.
*/
void color(BMP *image, vector<float> &features) {

    int vertical = image->TellHeight() / 8;
    int horizontal = image->TellWidth() / 8;

    double sum_R, sum_G, sum_B;

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {

            sum_R = 0;
            sum_G = 0;
            sum_B = 0;

            for (int k = 0; k < vertical; k++) {
                for (int m = 0; m < horizontal; m++) {

                    sum_R += (*image)(horizontal * j + m, vertical * i + k)->Red;
                    sum_G += (*image)(horizontal * j + m, vertical * i + k)->Green;
                    sum_B += (*image)(horizontal * j + m, vertical * i + k)->Blue;

                }
            }

            sum_R /= (vertical * horizontal * 255.0);
            sum_G /= (vertical * horizontal * 255.0);
            sum_B /= (vertical * horizontal * 255.0);

            features.push_back(sum_R);
            features.push_back(sum_G);
            features.push_back(sum_B);
        }
    }
}