#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"

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

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

//================================================================================


#define IMAGE_SIZE 64
#define WIDTH 16
#define HEIGHT 8
#define SEGMENT 16
#define PI 3.1415962

void grayscale(BMP *image, Matrix<float> &matrix) {

    Rescale(*image , 'F', IMAGE_SIZE);
    
    int height = image->TellHeight();
    int width = image->TellWidth();

    if (height > width) {
        int border = (height - width) / 2;
        // initialize borders
        for (int i = 0; i < IMAGE_SIZE; i++) {
            for(int j = 0; j <= border; j++) {
                matrix(i, j) = 1;
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
                matrix(i, j) = 1;
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

void abs_and_angles(const Matrix<float> &sobel_x, const Matrix<float> &sobel_y, Matrix<float> &abs, Matrix<float> &angles) {
    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            abs(i, j) = sqrt(sobel_x(i, j) * sobel_x(i, j) + sobel_y(i, j) * sobel_y(i, j));
            angles(i, j) = std::atan2(sobel_y(i,j), sobel_x(i, j)) + PI;
        }
    }
}

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

void HOG(vector<float> &features, Matrix<float> &gray, Matrix<float> &sobel_x, Matrix<float> &sobel_y, Matrix<float> &abs, Matrix<float> &angels) {
    sobel(gray, sobel_x, sobel_y);
    abs_and_angles(sobel_x, sobel_y, abs, angels);
    calc_histogram(abs, angels, features, HEIGHT, WIDTH);
}

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


// Exatract features from dataset.
// You should implement this function by yourself =)
void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {

    Matrix<float>    gray(IMAGE_SIZE, IMAGE_SIZE);
    Matrix<float> sobel_x(IMAGE_SIZE, IMAGE_SIZE);
    Matrix<float> sobel_y(IMAGE_SIZE, IMAGE_SIZE);
    Matrix<float>     abs(IMAGE_SIZE, IMAGE_SIZE);
    Matrix<float>  angels(IMAGE_SIZE, IMAGE_SIZE);

    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {

        vector<float> one_image_features;

        grayscale(data_set[image_idx].first, gray);

        HOG(one_image_features, gray, sobel_x, sobel_y, abs, angels);
        LBP(gray, one_image_features);
        color(data_set[image_idx].first, one_image_features);

        features->push_back(make_pair(one_image_features, data_set[image_idx].second));
    }
}

//========================================================================================

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}