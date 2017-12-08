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

    bool ProcessHatThings(cv::Mat& frame);

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

    bool  ProcessEye(cv::Mat& in);

    const vector<Rect>& getFaces()
    {
    	return faces;
    }

    bool ProcessFaceBeautification(cv::Mat& frame);

    void setDetScale(double num)
    {
    	m_Scale = num;
    }

    void setEyesDetScale(double num)
    {
        m_eyeDecScale = num;
    }

    void MaxFrame(Mat& frame ,double strength = 0.9, double radius = 0.0);
private:
    void initDetectors(CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2);

    void MaxFrame(IplImage* frame);
    void MinFrame(IplImage* frame);
    /*
	int prepareToProcessFaceDet();
	void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip );*/

    Rect faceRect;  // 检测出来的人脸的位置
    Rect searchedLeftEye, searchedRightEye; //左右眼检测
    Point leftEye, rightEye;    // 标记检测出来的眼睛
    bool gotFaceAndEyes ;

	Mat myMat;
	Mat m_Hat;
	int cvAdd4cMat_q(cv::Mat &dst, cv::Mat &scr, double scale) ;
	bool isNeedToBeUpdate;

	string cascadeName;
	string nestedCascadeName;
/*
	CascadeClassifier cascade;
	CascadeClassifier nestedCascade;
*/
    CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade1;
    CascadeClassifier eyeCascade2;

    unsigned int m_eyesStaticCount;
    unsigned int  m_faceStaticCount;

    double m_rtHat ;

	vector<Rect> faces,faces2;
    Point m_eyesWeight;
	cv::Mat* m_dectedPreMat;
	double m_Scale;
    double m_eyeDecScale;
};
#endif