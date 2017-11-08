#include "EXTfunction.h"

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
#include <pthread.h>
#include "vibe-background-sequential.h"
#include <pthread.h>

using namespace cv;
using namespace std;

EXTfunction::EXTfunction()
{
	//to do
	myMat= imread("frameBox3.png",-1); 
	//imageRotate(myMat, myMat, 30, false);

	m_Hat= imread("hat2.png",-1);
	imageRotate(m_Hat, m_Hat, 10, false);

	if(prepareToProcessFaceDet())
	{
		cerr << "WARNING: classifier filed to load" << endl;
	}
}

EXTfunction::~EXTfunction(){
	// to do
}

//图像旋转: src为原图像， dst为新图像, angle为旋转角度, isClip表示是采取缩小图片的方式  
int EXTfunction::imageRotate(InputArray src, OutputArray dst, double angle, bool isClip)  
{  
    Mat input = src.getMat();  
    if( input.empty() ) {  
        return -1;  
    }  
  
    //得到图像大小  
    int width = input.cols;  
    int height = input.rows;  
  
    //计算图像中心点  
    Point2f center;  
    center.x = width / 2.0;  
    center.y = height / 2.0;  
  
    //获得旋转变换矩阵  
    double scale = 1.0;  
    Mat trans_mat = getRotationMatrix2D( center, -angle, scale );  
  
    //计算新图像大小  
    double angle1 = angle  * CV_PI / 180. ;  
    double a = sin(angle1) * scale;  
    double b = cos(angle1) * scale;  
    double out_width = height * fabs(a) + width * fabs(b); //外边框长度  
    double out_height = width * fabs(a) + height * fabs(b);//外边框高度  
  
    int new_width, new_height;  
    if ( ! isClip ) {  
        new_width = cvRound(out_width);  
        new_height = cvRound(out_height);  
    } else {  
        //calculate width and height of clip rect  
        double angle2 = fabs(atan(height * 1.0 / width)); //即角度 b  
        double len = width * fabs(b);  
        double Y = len / ( 1 / fabs(tan(angle1)) + 1 / fabs(tan(angle2)) );  
        double X = Y * 1 / fabs(tan(angle2));  
        new_width = cvRound(out_width - X * 2);  
        new_height= cvRound(out_height - Y * 2);  
    }  
  
    //在旋转变换矩阵中加入平移量  
    trans_mat.at<double>(0, 2) += cvRound( (new_width - width) / 2 );  
    trans_mat.at<double>(1, 2) += cvRound( (new_height - height) / 2);  
  
    //仿射变换  
    warpAffine( input, dst, trans_mat, Size(new_width, new_height));  
  
    return 0;  
}  
    
bool EXTfunction::ProcessFrameBox(cv::Mat* in)
{
    // to do
	Size dsize = Size(in->cols,in->rows);
	resize(myMat, myMat,dsize);

	Mat img1_t1(*in, cvRect(0, 0, myMat.cols, myMat.rows));  
	cvAdd4cMat_q(img1_t1,myMat,1.0);  
	//imshow("Resoult2", img1_t1);
	
	//*in = *in + myMat;
    return true;
}

bool EXTfunction::ProcessHatThings(cv::Mat& frame)
{
	for ( size_t i = 0; i < faces.size(); i++ )
    {
    	//cout << i << ":" << faces[i]  << endl;
        Size dsize = Size(faces[i].width,faces[i].height);
        resize(m_Hat, m_Hat,dsize);

        Mat newMat(frame, faces[i]); 
        cvAdd4cMat_q(newMat,m_Hat,1.0);  
        //cv::Mat imageROI= frame(faces[i]);
        //m_Hat.copyTo(imageROI);
    }
    imshow("hello",m_Hat);

    return true;
}

int EXTfunction::cvAdd4cMat_q(cv::Mat &dst, cv::Mat &scr, double scale)    
{    
    if (dst.channels() != 3 || scr.channels() != 4)    
    {    
        return true;    
    }    
    if (scale < 0.01)    
        return false;    
    std::vector<cv::Mat>scr_channels;    
    std::vector<cv::Mat>dstt_channels;    
    split(scr, scr_channels);    
    split(dst, dstt_channels);    
    CV_Assert(scr_channels.size() == 4 && dstt_channels.size() == 3);    
  
    if (scale < 1)    
    {    
        scr_channels[3] *= scale;    
        scale = 1;    
    }    
    for (int i = 0; i < 3; i++)    
    {    
        dstt_channels[i] = dstt_channels[i].mul(255.0 / scale - scr_channels[3], scale / 255.0);    
        dstt_channels[i] += scr_channels[i].mul(scr_channels[3], scale / 255.0);    
    }    
    merge(dstt_channels, dst);    
    return true;    
}   

int EXTfunction::prepareToProcessFaceDet()
{
	cascadeName = "./haarcascade_frontalface_alt.xml";
    nestedCascadeName = "./haarcascade_eye_tree_eyeglasses.xml";
    m_Scale = 1;

  	if ( !nestedCascade.load(nestedCascadeName))
  	{
        cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
        return -1;
  	}
    if(!cascade.load( cascadeName ))
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }
    return 0;
}

void EXTfunction::detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip )
{
    double t = 0;
    //vector<Rect> faces, faces2;
    faces.clear();
    faces2.clear();
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
    
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        //Point center;
        //Scalar color = colors[i%8];
        int radius;

        /*double aspect_ratio = (double)r.width/r.height;
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
                       color, 3, 8, 0);*/
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
        /*for ( size_t j = 0; j < nestedObjects.size(); j++ )
        {
            Rect nr = nestedObjects[j];
            center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
            center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
            radius = cvRound((nr.width + nr.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }*/
    }

    t = (double)getTickCount() - t;
    //printf( "detection time = %g ms\n", t*1000/getTickFrequency());
    //imshow( "result", img );
} 

bool EXTfunction::ProcessFaceDetect(cv::Mat* in)
{
	changeDetMat(in);
	detectAndDraw( *m_dectedPreMat, cascade, nestedCascade, m_Scale , false );

	return true;
}

bool EXTfunction::ProcessFaceBeautification(cv::Mat& frame)
{
	int KERNEL_SIZE = 31;  
	
    for ( size_t i = 0; i < faces.size(); i++ )
    {
    	cout << i << ": " << faces[i]  << endl;
        int radius;
        Mat image_ori = frame(faces[i]);
        Mat frameBfBil = image_ori.clone();
	    for (int i = 1; i < KERNEL_SIZE; i = i + 2)  
	    {  
	        bilateralFilter(frameBfBil,image_ori,i,i*2,i/2);  
	    } 
    }
    return true; 
}