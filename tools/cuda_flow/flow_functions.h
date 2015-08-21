
#ifndef FLOW_FUNCTIONS_H
#define FLOW_FUNCTIONS_H

//******************************************************************************
//******************************************************************************

//=============================================================================
// System Includes
//=============================================================================
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>

//=============================================================================
// OPENCV Includes
//=============================================================================
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//******************************************************************************
//******************************************************************************

// Declaration of functions
void hsv2rgb(float h, float s, float v, unsigned char &r, unsigned char &g, unsigned char &b);
void drawMotionField(cv::Mat &imgU, cv::Mat &imgV, cv::Mat &imgMotion,
                     int xSpace, int ySpace, float cutoff, float multiplier, CvScalar color);
void drawLegendHSV(cv::Mat &imgColor, int radius, int cx, int cy);
void drawColorField(cv::Mat &imgU, cv::Mat &imgV, cv::Mat &imgColor);

//******************************************************************************
//******************************************************************************

#endif // FLOW_FUNCTIONS_H
