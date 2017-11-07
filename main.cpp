#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio>
#include <assert.h>
#include <time.h>
#include "TODynamiacBackgroundExtraction.h"
#include "EXTfunction.h"

int frameNumber = 1; 

#define RESULATION_FPS 30
#define RESULATION_DIS 2048

using namespace cv;
using namespace std;

string cascadeName;
string nestedCascadeName;

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip )
{
    double t = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors[] =
    {
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };
    Mat gray, smallImg;

    cvtColor( img, gray, COLOR_BGR2GRAY );
    double fx = 1 / scale;
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    t = (double)getTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE,
        Size(30, 30) );
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                 1.1, 2, 0
                                 //|CASCADE_FIND_BIGGEST_OBJECT
                                 //|CASCADE_DO_ROUGH_SEARCH
                                 |CASCADE_SCALE_IMAGE,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)getTickCount() - t;
    printf( "detection time = %g ms\n", t*1000/getTickFrequency());
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;

        double aspect_ratio = (double)r.width/r.height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
                       cvPoint(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                       color, 3, 8, 0);
        if( nestedCascade.empty() )
            continue;
        smallImgROI = smallImg( r );
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
            1.1, 2, 0
            //|CASCADE_FIND_BIGGEST_OBJECT
            //|CASCADE_DO_ROUGH_SEARCH
            //|CASCADE_DO_CANNY_PRUNING
            |CASCADE_SCALE_IMAGE,
            Size(30, 30) );
        for ( size_t j = 0; j < nestedObjects.size(); j++ )
        {
            Rect nr = nestedObjects[j];
            center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
            center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
            radius = cvRound((nr.width + nr.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
    }
    imshow( "result", img );
}

void demoHelp()
{
	printf("this demo is just for main function test\n");
	printf("1.vibe moving object get\n");
	printf("2.image add to image cofig\n");
	printf("3.vibe with facetected and hat location\n");
	printf("4.eyes get and be big\n");
	printf("5.beuty skin\n");
}

//去内轮廓
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

int main(int argc, char* argv[])
{
	/* Create GUI windows. */
	demoHelp();

	cascadeName = "./haarcascade_frontalface_alt.xml";
    nestedCascadeName = "./haarcascade_eye_tree_eyeglasses.xml";
    double scale = 1;

    CascadeClassifier cascade, nestedCascade;
  	if ( !nestedCascade.load( nestedCascadeName ) )
        cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }

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
    Mat res;
    int sobelThreshod = 2;

	TODynamiacBackgroundExtraction m_TOobject;
	m_TOobject.setNeedToBeUpdate(false);

	EXTfunction m_extension;

	VideoCapture cap(0);
	if(!cap.isOpened())
	{
		return -1;
	}

	while((char)keyboard != 'q' && (char)keyboard != 27)
	{
		cap >> frameOrg;
		if(frameOrg.cols <= 1 || frameOrg.rows <= 1)
		{
			continue;
		}

		//skip 10 frames to make data stable.
		static int begin_frame_count = 1;
		if(begin_frame_count < 10)
		{
			begin_frame_count++;
			continue;
		}
	    Size dsize = Size(640,480);
	    resize(frameOrg, frame,dsize);

	    double times = (double)getTickCount();
	    
#if 0 //VIBE
	    //sobel 法高效的取得轮廓
		Sobel(frame, XYImage, CV_16S, 1, 1, 2 * sobelThreshod + 1, 1, 10,BORDER_REPLICATE);  
        convertScaleAbs(XYImage, XYImage);  
        cvtColor(XYImage, XYImageG, CV_BGR2GRAY);
        
        XYImageG.convertTo(XYImageG,CV_8U,1.0);
        threshold(XYImageG,XYImageG,30,255,CV_THRESH_BINARY );

        //形态闭操作
        dilate(XYImageG,XYImageG,element_1);
        erode(XYImageG,XYImageG,element_1);
		m_TOobject.ProcessVideo(&frame, &segmentationMap, frameNumber);

		//次方形态闭
		dilate(segmentationMap,segmentationMap,element);
		dilate(XYImageG,XYImageG,element);
		erode(XYImageG & segmentationMap,res,element);
		
		IplImage iplimg = res;
		//去内轮廓
		FillInternalContours(&iplimg,40000);
		//imshow("ResoultG", XYImageG);
		//imshow("ResoultB", res);
		cvtColor(res, res, CV_GRAY2BGR);

		//轮廓和vibe结果与
		frame = res & frame;
        imshow("VIBE Resoult", frame);
#else
        //imshow("VIBE Resoult", frame);
#endif
        //detectAndDraw( frame, cascade, nestedCascade, scale , false );

		//m_extension.ProcessFrameBox(&frame);

		imshow("Resault1", frame);

		int KERNEL_SIZE = 31;  
		Mat frameBfBil = frame.clone();
	    for (int i = 1; i < KERNEL_SIZE; i = i + 2)  
	    {  
	        bilateralFilter(frameBfBil,frame,i,i*2,i/2);  
	    }  
		
		imshow("Resault", frame);
		++frameNumber;

		times = (double)getTickCount() - times;
    	printf( "Running time is: %g ms.\n", times*1000/getTickFrequency());
		keyboard = waitKey(1);
	}

	cap.release();
	destroyAllWindows();
	return 0;
}
