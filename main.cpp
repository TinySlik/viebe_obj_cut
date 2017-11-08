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
#include <pthread.h>

using namespace std;
#define NUM_THREADS 1

int frameNumber = 1; 

#define RESULATION_FPS 30
#define RESULATION_DIS 2048

using namespace cv;
using namespace std;

string cascadeName;
string nestedCascadeName;

// 线程的运行函数,函数返回的是函数指针，便于后面作为参数  
void* say_hello(void* args)
{
    cout << "Hello Runoob！" << endl;
    return NULL;
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
	m_extension.setDetScale(2);

	VideoCapture cap(0);
	if(!cap.isOpened())
	{
		return -1;
	}

	long long sumTime = 0;
	// 定义线程的 id 变量，多个变量使用数组
    pthread_t tids[NUM_THREADS];
    for(int i = 0; i < NUM_THREADS; ++i)
    {
        //参数依次是：创建的线程id，线程参数，调用的函数，传入的函数参数
        int ret = pthread_create(&tids[i], NULL, say_hello, NULL);
        if (ret != 0)
        {
           cout << "pthread_create error: error_code=" << ret << endl;
        }
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
	    Size dsize = Size(320*2,240*2);
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
        //face dect
        m_extension.ProcessFaceDetect(&frame);

        
#if 1   //skin beauty
		m_extension.ProcessFaceBeautification(frame);
#endif	

		m_extension.ProcessHatThings(frame);

		//hat and box
		m_extension.ProcessFrameBox(&frame);

		imshow("Res", frame);
		++frameNumber;

		times = (double)getTickCount() - times;
		sumTime += times*1000/getTickFrequency();
    	//printf( "Running time is: %g ms.\n", times*1000/getTickFrequency());
    	printf( "avg cost time is: %lld ms.\n", sumTime/frameNumber);
		keyboard = waitKey(1);
	}

	//等各个线程退出后，进程才结束，否则进程强制结束了，线程可能还没反应过来；
    pthread_exit(NULL);
	cap.release();
	destroyAllWindows();
	return 0;
}
