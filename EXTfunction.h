#ifndef EXT_FUNCTION_H
#define EXT_FUNCTION_H
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "vibe-background-sequential.h"
#include <opencv2/objdetect.hpp>

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

    void changeDetMat(cv::Mat* src)
    {
    	//to do ptherad safe things.
    	m_dectedPreMat = src;
    }

    bool ProcessFaceDetect(cv::Mat* in);

    const vector<Rect>& getFaces()
    {
    	return faces;
    }

    bool ProcessFaceBeautification(cv::Mat& frame);
    
private:

	int prepareToProcessFaceDet();
	void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip );
	Mat myMat;
	int cvAdd4cMat_q(cv::Mat &dst, cv::Mat &scr, double scale) ;
	bool isNeedToBeUpdate;

	string cascadeName;
	string nestedCascadeName;

	CascadeClassifier cascade;
	CascadeClassifier nestedCascade;

	vector<Rect> faces,faces2;
	cv::Mat* m_dectedPreMat;
	double m_Scale;
};
#endif