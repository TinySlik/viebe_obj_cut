#ifndef MYVIBE_INCLUDED
#define MYVIBE_INCLUDED

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "vibe-background-sequential.h"

using namespace std;
using namespace cv;

class myVibe
{
public:
    myVibe();
    ~myVibe();
    bool ProcessVideo(cv::Mat* inputFrame, cv::Mat* segmentationMap, long frameNumber);
private:
    vibeModel_Sequential_t *model; /* Model used by ViBe. */
};

#endif // MYVIBE_INCLUDED
