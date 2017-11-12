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

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"

//#include "../include/matrix.h"
#include "../include/Timer.h"
#include "../include/mycode.h"


using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

/**
@mainpage OBJECT CLASSIFICATION.
This project was made to solve classification problem.
@author This project was created by Alexander Tsygankov from 321 group.
*/

/**
@file task2.cpp
*/

using CommandLineProcessing::ArgvParser;


/// set of data which stores pointer to BMP image and its label
typedef vector<pair<BMP*, int> > TDataSet;
/// list of image filenames and its labels
typedef vector<pair<string, int> > TFileList;
/// set of data which stores image features vector and its labels
typedef vector<pair<vector<float>, int> > TFeatures;


/**
@function LoadFileList
load list of files and its labels
@param data_file is input filename of files and its labels.
@param file_list is output list of filenames.
*/
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

/**
@function LoadImage
load image and save them in dataset
@param file_list is list of files for loading.
@param data_set is output set of images and its labels.
*/
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

/**
@function SavePredictions
save result of prediction to file
@param file_list is list filenames and its labels.
@param labels is list of labels.
@param prediction_file is predictions filename.
*/
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



/**
@function ExtractFeutures
extracting features from image
@param data_set is input set of data which include BMP image pointer and its label.
@param features is output vector for features.
*/
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

/**
@function ClearDataset
clear dataset structure
@param data_set is set of data which will be cleaned.
*/
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

/**
@function TrainClassifier
train SVM classifier and save trained model
@param data_file is input data filename for classifier.
@param model_file is output filename for model.
*/
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

/**
@function PredictData
predict data and save predictions
@param data_file is input data filename to predict.
@param model_file is model filename.
@param prediction_file is output filename for predictions.
*/
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

/*

void test() {

    Matrix<float>    gray(IMAGE_SIZE, IMAGE_SIZE);
    Matrix<float> sobel_x(IMAGE_SIZE, IMAGE_SIZE);
    Matrix<float> sobel_y(IMAGE_SIZE, IMAGE_SIZE);
    Matrix<float>     abs(IMAGE_SIZE, IMAGE_SIZE);
    Matrix<float>  angles(IMAGE_SIZE, IMAGE_SIZE);

    BMP *image = new BMP();
    image->ReadFromFile("../../data/Lenna.bmp");
    const int NUM_ITER = 1000;
    Timer t;

    grayscale(image, gray);
    sobel(gray, sobel_x, sobel_y);

    t.start();
    for (auto idx = 0; idx < NUM_ITER; ++idx)
        abs_and_angles(sobel_x, sobel_y, abs, angles);
    t.check("not SSE");

    t.restart();
    for (auto idx = 0; idx < NUM_ITER; ++idx)
        abs_and_anglesSSE(sobel_x, sobel_y, abs, angles);
    t.check("SSE");
}
*/

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2017.");
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

/*
    cout << "TEST" << endl;
    test();
    cout << "TEST" << endl;
*/

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