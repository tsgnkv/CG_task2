#include "gtest/gtest.h"
#include "mycode.h"
#include "EasyBMP.h"

/**
@file abs_test.cpp
*/

/**
@function TEST
check equality of two matrix abs and absSSE
*/
TEST(test, checkSSE) {

	SetEasyBMPwarningsOff();
    BMP* image = new BMP();

    image->ReadFromFile("../../data/Lenna.bmp");

    Matrix<float>    gray(IMAGE_SIZE, IMAGE_SIZE);
    Matrix<float> sobel_x(IMAGE_SIZE, IMAGE_SIZE);
    Matrix<float> sobel_y(IMAGE_SIZE, IMAGE_SIZE);
    Matrix<float>     abs(IMAGE_SIZE, IMAGE_SIZE);
    Matrix<float>     absSSE(IMAGE_SIZE, IMAGE_SIZE);
    Matrix<float>  angles(IMAGE_SIZE, IMAGE_SIZE);

    grayscale(image, gray);
    sobel(gray, sobel_x, sobel_y);

    abs_and_angles(sobel_x, sobel_y, abs, angles);
    abs_and_anglesSSE(sobel_x, sobel_y, absSSE, angles);

    for (uint i = 0; i < IMAGE_SIZE; i++) {
        for (uint j = 0; j < IMAGE_SIZE; j++) {
            EXPECT_NEAR(abs(i, j), absSSE(i, j), 0.00001);
        }
    }

    delete image;
}

/**
@function main
is main function for testing system
*/
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

