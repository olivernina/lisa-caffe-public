
//=============================================================================
//
// brox_flow.cpp
// Main file for testing OpenCV GPU Brox Optical Flow
// Author: Pablo F. Alcantarilla
// Institution: ALCoV, Université d'Auvergne
// Date: 23/11/2012
// Email: pablofdezalc@gmail.com
//=============================================================================

#include "brox_flow.h"

// Namespaces
using namespace std;
using namespace cv;
using namespace cv::gpu;

// Some global variables for the optical flow
const float alpha_ = 0.12;
const float gamma_ = 5;
const float scale_factor_ = 0.9;
const int inner_iterations_ = 3;
const int outer_iterations_ = 50;
const int solver_iterations_ = 20;
const bool resize_img = false;
const float rfactor = 2.0;

//******************************************************************************
//******************************************************************************

/** Main Function */
int main( int argc, char *argv[] )
{
    // Variables for CUDA Brox Optical flow
    GpuMat frame0GPU, frame1GPU, uGPU, vGPU;
    Mat frame0_rgb_, frame1_rgb_, frame0_rgb, frame1_rgb, frame0, frame1;
    Mat frame0_32, frame1_32, imgU, imgV;
    Mat motion_flow, flow_rgb;
    int nframes = 0, width = 0, height = 0;
    char cad[NMAX_CHARACTERS];

    // Variables for measuring computation times
    struct timeval tod1;
    double t1 = 0.0, t2 = 0.0, tdflow = 0.0, tvis = 0.0;

    // Check input arguments
    if( argc != 2 )
    {
        cout << "Error introducing input arguments!!" << endl;
        cout << "Number of input arguments: " << argc << endl;
        cout << "The format is: ./brox_flow video_file.avi" << endl;
        return -1;
    }

    // Show CUDA information
    cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

    // Create OpenCV windows
    //namedWindow("Dense Flow",CV_WINDOW_OPENGL);
    namedWindow("Dense Flow",CV_WINDOW_NORMAL);

    // Create the optical flow object
    cv::gpu::BroxOpticalFlow dflow(alpha_,gamma_,scale_factor_,inner_iterations_,outer_iterations_,solver_iterations_);

    // Open the video file
    VideoCapture cap(argv[1]);
    if( cap.isOpened() == 0 )
    {
        return -1;
    }

    cap >> frame1_rgb_;

        frame1_rgb = cv::Mat(Size(frame1_rgb_.cols,frame1_rgb_.rows),CV_8UC3);
        width = frame1_rgb.cols;
        height = frame1_rgb.rows;
        frame1_rgb_.copyTo(frame1_rgb);

    // Convert the image to grey and float

            // Visualizatio
        while(frame1_rgb_.empty()==false)
	{
        imshow("Dense Flow",frame1_rgb);
	//frame1_rgb_.copyTo(frame1_rgb_);
	waitKey(30);
	cap>>frame1_rgb_;
	frame1_rgb_.copyTo(frame1_rgb);
	}
    // Destroy the windows
	destroyAllWindows();
    return 0;
}


