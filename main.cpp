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

void search_target(Mat mask_copy,Mat frame_copy,int area_min );
void showHelpInformation();
void closed_operation(Mat& src,Mat& dst,const Mat& element,const int times = 1);
void FillInternalContours(IplImage *pBinary, double dAreaThre);

void showHelpInformation()
{
    cout
    << "--------------------------------------------------------------------------" << endl
    << "This program shows how to use ViBe with OpenCV                            " << endl
    << "Usage:"                                                                     << endl
    << "--------------------------------------------------------------------------" << endl
    << endl;
}

/* Sobel template
a00 a01 a02
a10 a11 a12
a20 a21 a22
*/
void MySobel(IplImage* gray, IplImage* gradient)
{
	unsigned char a00, a01, a02;
	unsigned char a10, a11, a12;
	unsigned char a20, a21, a22;
	CvScalar color ;
	for (int i=1; i<gray->height-1; ++i)
	{
		for (int j=1; j<gray->width-1; ++j)
		{
			a00 = cvGet2D(gray, i-1, j-1).val[0];
			a01 = cvGet2D(gray, i-1, j).val[0];
			a02 = cvGet2D(gray, i-1, j+1).val[0];
			a10 = cvGet2D(gray, i, j-1).val[0];
			a11 = cvGet2D(gray, i, j).val[0];
			a12 = cvGet2D(gray, i, j+1).val[0];
			a20 = cvGet2D(gray, i+1, j-1).val[0];
			a21 = cvGet2D(gray, i+1, j).val[0];
			a22 = cvGet2D(gray, i+1, j+1).val[0];
			// x方向上的近似导数  卷积运算
			double ux = a20 * (1) + a10 * (2) + a00 * (1) + (a02 * (-1) + a12 * (-2) + a22 * (-1));
			// y方向上的近似导数  卷积运算
			double uy = a02 * (1) + a01 * (2) + a00 * (1) + a20 * (-1) + a21 * (-2) + a22 * (-1);
			color.val[0] = sqrt(ux*ux + uy*uy);
			cvSet2D(gradient, i, j, color);
		}
	}
}

int main(int argc, char* argv[])
{
	/* Create GUI windows. */
  	namedWindow("Frame");
  	namedWindow("Segmentation by ViBe");

	/* Variables. */
  	static int frameNumber = 1; /* The current frame number */
	Mat frame;                  /* Current frame. */
	Mat frameGray;
	Mat segmentationMap;        /* Will contain the segmentation map. This is the binary output map. */
	int keyboard = 0;           /* Input from keyboard. Used to stop the program. Enter 'q' to quit. */
	Mat dstImage, XImage, YImage, XYImage ,XYImageG;  
    Mat AbsXImage, AbsYImage, AbsXYImage;  
    Mat element = getStructuringElement(MORPH_RECT,Size(15,15));
    Mat element5 = getStructuringElement(MORPH_RECT,Size(5,5));
    Mat element30 = getStructuringElement(MORPH_ELLIPSE,Size(30,30));
    int sobelThreshod = 2;

	myVibe m_vibe;
	VideoCapture cap(0);
	if(!cap.isOpened())
	{
		return -1;
	}
	while((char)keyboard != 'q' && (char)keyboard != 27)
	{
		cap >> frame;

		Mat orgIm = imread("theImage.png");  
	    int extRows = 20;  
	    int extCols = 20;  
	    copyMakeBorder( frame, frame, extRows, extRows, extCols, extCols, BORDER_REPLICATE);

		Sobel(frame, XImage, CV_16S, 1, 0, 2 * sobelThreshod + 1, 1, 1);  
        convertScaleAbs(XImage, AbsXImage);  
        Sobel(frame, YImage, CV_16S, 0, 1, 2 * sobelThreshod + 1, 1, 1);  
        convertScaleAbs(YImage, AbsYImage);  
        addWeighted(AbsXImage, 0.5, AbsYImage, 0.5, 0, XYImage);

		m_vibe.ProcessVideo(&frame, &segmentationMap, frameNumber);

		/* Shows the current frame and the segmentation map. */
		//imshow("Frame", frame);

        cvtColor(XYImage, XYImageG, CV_BGR2GRAY);
        XYImageG.convertTo(XYImage,CV_8U,225.0/RESULATION_DIS);

		//imshow("Segmentation by ViBe", segmentationMap);
		//imshow("XYImage", XYImage);
		Mat res = XYImage & segmentationMap;
		
		closed_operation(res,res,element);
		//IplImage* iplimg;
		//*iplimg = IplImage(res);
		cv::Mat img2;
		IplImage iplimg = res;
		FillInternalContours(&iplimg,100);
		//FillInternalContours(&iplimg,200);
		//closed_operation(res,res,element30);
		//FillInternalContours(&iplimg,120);
		//grabCut(frame,)
		//Mat resS = cvarrToMat(iplimg);
		//Mat resS = Mat(iplimg);

		//thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
		//Mat resS;
		//threshold(res,resS,200,255,CV_THRESH_TOZERO_INV );
		//imshow("frame",frame);
		cvtColor(res, res, CV_GRAY2BGR);
		erode(res,res,element5);
		res = res & frame;
        imshow("Resoult", res);
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

void search_target(Mat mask_copy,Mat frame_copy ,int area_min)
{
	int area_copy=area_min;
	cv::Mat bwImg= mask_copy;  
	vector<vector<cv::Point> > contours ;     
	cv::findContours(bwImg,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);  	// Ñ°ÕÒ×î´óÁ¬Í¨Óò  

	double maxArea =0,Area1 = 0,Area2= 0;  
	vector<cv::Point> maxContour,maxContour1,maxContour2;  
	size_t i1 = 0;
	double area1 = 0;
	size_t i2 = 0;
	double area2 = 0;

	int maxCount = 0;
	if(contours.size()>0)
	{
		int index_last = 0, index_max = 0,index_second=0;
		for(size_t i = 0; i < contours.size(); i++)  
		{  

			double area = cv::contourArea(contours[i]);
			if (area > maxArea)  
			{  
				index_second =index_last; 
				index_last = index_max;
				maxArea = area;  				
				index_max = i;
				maxContour = contours[i];
				maxContour1 = contours[index_last];
				maxContour2 = contours[index_second];
			}
		}

		cv::Mat result;  
		frame_copy.copyTo(result);   
		for (size_t ii = 0; ii < contours.size(); ii++)  
		{  
			cv::Rect r = cv::boundingRect(contours[ii]); 
			if( cv::contourArea(contours[ii])>area_min)
			{
				cv::rectangle(result, r, cv::Scalar(255)); 
			}
		}  
		cv::imshow("GG1", result);
		//cv::Rect maxRect  = cv::boundingRect(maxContour);
		//cv::Rect rmaxRect1 = cv::boundingRect(contours[index_last]);
		//cv::Rect rmaxRect2 = cv::boundingRect(contours[index_second]); 
		//cv::Rect maxRect  = cv::boundingRect(maxContour);
		//cv::rectangle(result2, maxRect,  cv::Scalar(255)); 
		//cv::rectangle(result2, rmaxRect1, cv::Scalar(255)); 
		//cv::rectangle(result2, rmaxRect2, cv::Scalar(255)); 
	}
	else 
	{
		cv::imshow("GG3", frame_copy); 
	}
}