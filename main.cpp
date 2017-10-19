#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio>
#include <assert.h>
#include <time.h>
#include <iostream>
#include "myVibe.h"

//using namespace openni;
int GRAY_TH=0;//ÊäÈëÍ¼Ïñ·§Öµ»¯ *
int AREA_TH=0;//×îÐ¡¿òÑ¡Ãæ»ý *
int LONG_TH=0;//ÊÇ·ñÐèÒªÖØ½¨ *
int TIME_TH=0;//Ç°¾°µãÁ¬Ðø³öÏÖµÄÊ±¼äºó±»ÖØ½¨Îª±³¾° *

int frameNumber = 1; //µ±Ç°Ö¡Êý

/*	#define RESULATION_X 224
	#define RESULATION_Y 172
*/
#define RESULATION_FPS 30
#define RESULATION_DIS 2048

using namespace cv;
using namespace std;

void closed_operation(Mat& src,Mat& dst,const Mat& element,const int times = 1);
void FillInternalContours(IplImage *pBinary, double dAreaThre);

int main(int argc, char* argv[])
{
	/* Create GUI windows. */
  	namedWindow("Frame");
  	namedWindow("Segmentation by ViBe");

	/* Variables. */
  	static int frameNumber = 1; /* The current frame number */
	Mat frame;                  /* Current frame. */
	Mat frameGray;
	Mat frameOrg;
	Mat segmentationMap;        /* Will contain the segmentation map. This is the binary output map. */
	int keyboard = 0;           /* Input from keyboard. Used to stop the program. Enter 'q' to quit. */
	Mat XYImage ,XYImageG;   
	Mat element_1 = getStructuringElement(MORPH_RECT,Size(5,5));
    Mat element = getStructuringElement(MORPH_RECT,Size(10,10));
    Mat element_3 = getStructuringElement(MORPH_CROSS,Size(30,30));
    int sobelThreshod = 2;

	myVibe m_vibe;
	VideoCapture cap(0);
	if(!cap.isOpened())
	{
		return -1;
	}
	while((char)keyboard != 'q' && (char)keyboard != 27)
	{
		cap >> frameOrg;
	    Size dsize = Size(frameOrg.cols*0.5,frameOrg.rows*0.5);
	    Mat frame ;
	    resize(frameOrg, frame,dsize);
		Sobel(frame, XYImage, CV_16S, 1, 1, 2 * sobelThreshod + 1, 1, 10,BORDER_REPLICATE);  

        convertScaleAbs(XYImage, XYImage);  
        cvtColor(XYImage, XYImageG, CV_BGR2GRAY);
        
        XYImageG.convertTo(XYImageG,CV_8U,1.0);
        threshold(XYImageG,XYImageG,30,255,CV_THRESH_BINARY );
        dilate(XYImageG,XYImageG,element_1);
        erode(XYImageG,XYImageG,element_1);
		m_vibe.ProcessVideo(&frame, &segmentationMap, frameNumber);

		Mat res = XYImageG & segmentationMap;
		
		imshow("Resoult1", res);
		dilate(res,res,element);
		erode(res,res,element);
		IplImage iplimg = res;
		FillInternalContours(&iplimg,300);
		imshow("Resoult2", res);
		
		//cvtColor(res, res, CV_GRAY2BGR);
		//res = res & frame;
        //imshow("Resoult", res);
		++frameNumber;
		keyboard = waitKey(1);
	}

	cap.release();
	destroyAllWindows();
	return 0;
}

void FillInternalContours(IplImage *pBinary, double dAreaThre)   
{   
    double dConArea;   
    CvSeq *pContour = NULL;   
    CvSeq *pConInner = NULL;   
    CvMemStorage *pStorage = NULL;   
    // 执行条件   
    if (pBinary)   
    {   
        // 查找所有轮廓   
        pStorage = cvCreateMemStorage(0);   
        cvFindContours(pBinary, pStorage, &pContour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);   
        // 填充所有轮廓   
        cvDrawContours(pBinary, pContour, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 2, CV_FILLED, 8, cvPoint(0, 0));  
        // 外轮廓循环   
        for (; pContour != NULL; pContour = pContour->h_next)   
        {   
            // 内轮廓循环   
            for (pConInner = pContour->v_next; pConInner != NULL; pConInner = pConInner->h_next)   
            {   
                // 内轮廓面积   
                dConArea = fabs(cvContourArea(pConInner, CV_WHOLE_SEQ));   
                if (dConArea <= dAreaThre)   
                {   
                    cvDrawContours(pBinary, pConInner, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 0, CV_FILLED, 8, cvPoint(0, 0));  
                }   
            }   
        }   
        cvReleaseMemStorage(&pStorage);   
        pStorage = NULL;   
    }   
}   

void closed_operation(Mat& src,Mat& dst,const Mat& element,const int times)
{
	for (int i = 0; i < times; ++i)
	{
		dilate(src,dst,element);
        erode(dst,dst,element);
	}
}
