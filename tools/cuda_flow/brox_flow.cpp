
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
    namedWindow("Dense Flow",CV_WINDOW_OPENGL);
    namedWindow("Motion Flow",CV_WINDOW_NORMAL);

    // Create the optical flow object
    cv::gpu::BroxOpticalFlow dflow(alpha_,gamma_,scale_factor_,inner_iterations_,outer_iterations_,solver_iterations_);

    // Open the video file
    VideoCapture cap(argv[1]);
    if( cap.isOpened() == 0 )
    {
        return -1;
    }

    cap >> frame1_rgb_;

    if( resize_img == true )
    {
        frame1_rgb = cv::Mat(Size(cvRound(frame1_rgb_.cols/rfactor),cvRound(frame1_rgb_.rows/rfactor)),CV_8UC3);
        width = frame1_rgb.cols;
        height = frame1_rgb.rows;
        cv::resize(frame1_rgb_,frame1_rgb,cv::Size(width,height),0,0,INTER_LINEAR);
    }
    else
    {
        frame1_rgb = cv::Mat(Size(frame1_rgb_.cols,frame1_rgb_.rows),CV_8UC3);
        width = frame1_rgb.cols;
        height = frame1_rgb.rows;
        frame1_rgb_.copyTo(frame1_rgb);
    }

    // Allocate memory for the images
    frame0_rgb = cv::Mat(Size(width,height),CV_8UC3);
    flow_rgb = cv::Mat(Size(width,height),CV_8UC3);
    motion_flow = cv::Mat(Size(width,height),CV_8UC3);
    frame0 = cv::Mat(Size(width,height),CV_8UC1);
    frame1 = cv::Mat(Size(width,height),CV_8UC1);
    frame0_32 = cv::Mat(Size(width,height),CV_32FC1);
    frame1_32 = cv::Mat(Size(width,height),CV_32FC1);

    // Convert the image to grey and float
    cvtColor(frame1_rgb,frame1,CV_BGR2GRAY);
    frame1.convertTo(frame1_32,CV_32FC1,1.0/255.0,0);

    while( frame1.empty() == false )
    {
        if( nframes >= 1 )
        {
            gettimeofday(&tod1,NULL);
            t1 = tod1.tv_sec + tod1.tv_usec / 1000000.0;

            // Upload images to the GPU
            frame1GPU.upload(frame1_32);
            frame0GPU.upload(frame0_32);

            // Do the dense optical flow
            dflow(frame0GPU,frame1GPU,uGPU,vGPU);

            uGPU.download(imgU);
            vGPU.download(imgV);

            gettimeofday(&tod1,NULL);
            t2 = tod1.tv_sec + tod1.tv_usec / 1000000.0;
            tdflow = 1000.0*(t2-t1);
        }

        if( nframes >= 1 )
        {
            gettimeofday(&tod1,NULL);
            t1 = tod1.tv_sec + tod1.tv_usec / 1000000.0;

            // Draw the optical flow results
            drawColorField(imgU,imgV,flow_rgb);

            frame1_rgb.copyTo(motion_flow);
            drawMotionField(imgU,imgV,motion_flow,15,15,.0,3,CV_RGB(0,255,0));

            // Visualization
            imshow("Dense Flow",flow_rgb);
            imshow("Motion Flow",motion_flow);

            waitKey(3);

            gettimeofday(&tod1,NULL);
            t2 = tod1.tv_sec + tod1.tv_usec / 1000000.0;
            tvis = 1000.0*(t2-t1);
        }

        // Save results
        if( SAVE_RESULTS == true )
        {
            sprintf(cad,"./output/motion/image%04d.jpg",nframes);
            imwrite(cad,motion_flow);

            sprintf(cad,"./output/flow/image%04d.jpg",nframes);
            imwrite(cad,flow_rgb);
        }

        // Set the information for the previous frame
        frame1_rgb.copyTo(frame0_rgb);
        cvtColor(frame0_rgb,frame0,CV_BGR2GRAY);
        frame0.convertTo(frame0_32,CV_32FC1,1.0/255.0,0);

        // Read the next frame
        nframes++;
        cap >> frame1_rgb_;

        if( frame1_rgb_.empty() == false )
        {
            if( resize_img == true )
            {
                cv::resize(frame1_rgb_,frame1_rgb,cv::Size(width,height),0,0,INTER_LINEAR);
            }
            else
            {
                frame1_rgb_.copyTo(frame1_rgb);
            }

            cvtColor(frame1_rgb,frame1,CV_BGR2GRAY);
            frame1.convertTo(frame1_32,CV_32FC1,1.0/255.0,0);
        }
        else
        {
            break;
        }

        cout << "Frame Number: " << nframes << endl;
        cout << "Time Dense Flow: " << tdflow << endl;
        cout << "Time Visualization: " << tvis << endl << endl;
    }

    // Destroy the windows
    destroyAllWindows();

    return 0;
}


