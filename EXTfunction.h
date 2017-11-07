#ifndef EXT_FUNCTION_H
#define EXT_FUNCTION_H
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "vibe-background-sequential.h"

using namespace std;
using namespace cv;

class EXTfunction
{
public:
    EXTfunction();
    ~EXTfunction();
    //bool ProcessVideo(cv::Mat* inputFrame, cv::Mat* segmentationMap);

    bool ProcessFrameBox(cv::Mat* in);

    int imageRotate(InputArray src, OutputArray dst, double angle, bool isClip);
    
    void setNeedToBeUpdate(bool is)
    {
    	isNeedToBeUpdate = is;
    }
private:
	Mat myMat;
	int cvAdd4cMat_q(cv::Mat &dst, cv::Mat &scr, double scale) ;
	bool isNeedToBeUpdate;
};
#endif