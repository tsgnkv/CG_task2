#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include <xmmintrin.h>
#include <smmintrin.h>
#include <mmintrin.h>
#include <immintrin.h>

#include "EasyBMP.h"

#include "../include/matrix.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

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

void grayscale(BMP *image, Matrix<float> &matrix);

void sobel(const Matrix<float> &src, Matrix<float> &sobel_x, Matrix<float> &sobel_y);

void sobelSSE(const Matrix<float> &src, Matrix<float> &sobel_x, Matrix<float> &sobel_y);

void abs_and_angles(const Matrix<float> &sobel_x, const Matrix<float> &sobel_y, Matrix<float> &abs, Matrix<float> &angles);

void abs_and_anglesSSE(Matrix<float> &sobel_x, Matrix<float> &sobel_y, Matrix<float> &abs, Matrix<float> &angles);

void calc_histogram(const Matrix<float> &abs, const Matrix<float> &angels, vector<float> &features, int h, int w);

void HOG(vector<float> &features, Matrix<float> &gray, Matrix<float> &sobel_x, Matrix<float> &sobel_y, Matrix<float> &abs, Matrix<float> &angels);

void LBP(Matrix<float> &gray, vector<float> &features);

void color(BMP *image, vector<float> &features);