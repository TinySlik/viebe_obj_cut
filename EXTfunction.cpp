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
#include "detectObject.h"
#include "preprocessFace.h"
#include <math.h>

using namespace cv;
using namespace std;

#define EYE_STATIC_THE 10
#define FACE_STATIC_THE 10

#define EYE_IN_FACE_WIDTH_PER  0.25
#define EYE_IN_FACE_HEIGHT_PER  0.25

// 设置期望的人脸维度，设置为70*70
const int faceWidth = 70;
const int faceHeight = faceWidth;

const bool preprocessLeftAndRightSeparately = true;   // 是否分别对左侧和右侧人脸进行处理的标志

// 级联分类器
const char *faceCascadeFilename = "cv_resource/haarcascade_frontalface_alt.xml";     // LBP face detector.
//const char *faceCascadeFilename = "C:/opencv/sources/data/lbpcascades/haarcascade_frontalface_alt_tree.xml";  // Haar face detector.
//const char *eyeCascadeFilename1 = "C:/opencv/sources/data/lbpcascades/haarcascade_lefteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename2 = "C:/opencv/sources/data/lbpcascades/haarcascade_righteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename1 = "C:/opencv/sources/data/lbpcascades/haarcascade_mcs_lefteye.xml";       // Good eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename2 = "C:/opencv/sources/data/lbpcascades/haarcascade_mcs_righteye.xml";       // Good eye detector for open-or-closed eyes.
const char *eyeCascadeFilename1 = "cv_resource/haarcascade_eye.xml";               // Basic eye detector for open eyes only.
const char *eyeCascadeFilename2 = "cv_resource/haarcascade_eye_tree_eyeglasses.xml"; // Basic eye detector for open eyes if they might wear glasses.


EXTfunction::EXTfunction()
{
    m_Scale = 1;
    m_eyeDecScale =  1;
    gotFaceAndEyes = false;
    m_rtHat  = 0;
	//to do
	myMat= imread("cv_resource/frameBox3.png",-1); 
	//imageRotate(myMat, myMat, 30, false);

	m_Hat= imread("cv_resource/hat2.png",-1);
	
    m_eyesStaticCount =  0;

    m_faceStaticCount = 0;

    m_eyesWeight = Point(1,1);
	/*if(prepareToProcessFaceDet())
	{
		cerr << "WARNING: classifier filed to load" << endl;
	}*/

    //载入XML分类器
    initDetectors(faceCascade, eyeCascade1, eyeCascade2);
}

EXTfunction::~EXTfunction(){
	// to do
}

void EXTfunction::MaxFrame(IplImage* frame)  
{  
    uchar* old_data = (uchar*)frame->imageData;  
    uchar* new_data = new uchar[frame->widthStep * frame->height];  
  
    int center_X = frame->width / 2;  
    int center_Y = frame->height / 2;  
    int radius = frame->height;  
    int newX = 0;  
    int newY = 0;  
  
    int real_radius = (int)(radius / 2.0);  
    for (int i = 0; i < frame->width; i++)  
    {  
        for (int j = 0; j < frame->height; j++)  
        {  
            int tX = i - center_X;  
            int tY = j - center_Y;  
  
            int distance = (int)(tX * tX + tY * tY);  
            if (distance < radius * radius)  
            {  
                newX = (int)((float)(tX) / 2.0);  
                newY = (int)((float)(tY) / 2.0);  
  
                newX = (int) (newX * (sqrt((double)distance) / real_radius));  
                newX = (int) (newX * (sqrt((double)distance) / real_radius));  
  
                newX = newX + center_X;  
                newY = newY + center_Y;  
  
                new_data[frame->widthStep * j + i * 3] = old_data[frame->widthStep * newY + newX * 3];  
                new_data[frame->widthStep * j + i * 3 + 1] =old_data[frame->widthStep * newY + newX * 3 + 1];  
                new_data[frame->widthStep * j + i * 3 + 2] =old_data[frame->widthStep * newY + newX * 3 + 2];  
            }  
            else  
            {  
                new_data[frame->widthStep * j + i * 3] =  old_data[frame->widthStep * j + i * 3];  
                new_data[frame->widthStep * j + i * 3 + 1] =  old_data[frame->widthStep * j + i * 3 + 1];  
                new_data[frame->widthStep * j + i * 3 + 2] =  old_data[frame->widthStep * j + i * 3 + 2];  
            }  
        }  
    }  
    memcpy(old_data, new_data, sizeof(uchar) * frame->widthStep * frame->height);  
    delete[] new_data;  
}  

void EXTfunction::MaxFrame(Mat& frame ,double strength, double radius)
{
    double center_X = frame.cols / 2.0;  
    double center_Y = frame.rows / 2.0;   

    if (radius < 0.001)
    {
        radius = frame.cols > frame.rows ? (frame.rows/2 - 1): (frame.cols/2 - 1);
    }

    int nr=frame.rows;
    // 将3通道转换为1通道
    int nl=frame.cols*frame.channels();

    if(2*radius > frame.cols || 2*radius > frame.rows)
    {
        cout  << "to smale img col "<< frame.cols  << "rows"   << frame.rows <<  "with radious" << radius <<  "to max frame done.\n" << endl;
        return ;
    }
    
    Mat newFrame  = frame.clone();
    cout  << "frame.cols"   << frame.cols << "frame.rows" << frame.rows  << endl;
    for(int i=0;i<frame.rows;i++)
    {
        for(int j=0;j<frame.cols;j++)
        {
            double distance =  sqrt((j  - center_X)*(j  - center_X)+(i - center_Y)*(i - center_Y));
            if(distance < radius)
            {
                int xx  = j - strength*(j - center_X)*(radius - distance)/radius;
                int yy = i - strength*(i - center_Y)*(radius - distance)/radius;
                frame.at<Vec3b>(i,j)[0]=newFrame.at<Vec3b>(yy , xx)[0];
                frame.at<Vec3b>(i,j)[1]=newFrame.at<Vec3b>(yy , xx)[1];
                frame.at<Vec3b>(i,j)[2]=newFrame.at<Vec3b>(yy , xx)[2];
            }
        }
    }
}
  
  
void EXTfunction::MinFrame(IplImage* frame)  
{  
    uchar* old_data = (uchar*)frame->imageData;  
    uchar* new_data = new uchar[frame->widthStep * frame->height];  
  
    int center_X = frame->width / 2;  
    int center_Y = frame->height / 2;  
  
    int radius = 0;  
    double theta = 0;  
    int newX = 0;  
    int newY = 0;  
  
    for (int i = 0; i < frame->width; i++)  
    {  
        for (int j = 0; j < frame->height; j++)  
        {  
            int tX = i - center_X;  
            int tY = j - center_Y;  
  
            theta = atan2((double)tY, (double)tX);  
            radius = (int)sqrt((double)(tX * tX) + (double) (tY * tY));  
            int newR = (int)(sqrt((double)radius) * 12);  
            newX = center_X + (int)(newR * cos(theta));  
            newY = center_Y + (int)(newR * sin(theta));  
  
            if (!(newX > 0 && newX < frame->width))  
            {  
                newX = 0;  
            }  
            if (!(newY > 0 && newY < frame->height))  
            {  
                newY = 0;  
            }  
  
            new_data[frame->widthStep * j + i * 3] = old_data[frame->widthStep * newY + newX * 3];  
            new_data[frame->widthStep * j + i * 3 + 1] =old_data[frame->widthStep * newY + newX * 3 + 1];  
            new_data[frame->widthStep * j + i * 3 + 2] =old_data[frame->widthStep * newY + newX * 3 + 2];  
        }  
    }  
    memcpy(old_data, new_data, sizeof(uchar) * frame->widthStep * frame->height);  
    delete[] new_data;  
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
    if(!gotFaceAndEyes)
    {
        return false;
    }
    if(faceRect.width   < 10 || faceRect.height < 10)
    {
        return false;
    }

    Mat ne ;
    //Mat ne = imread("hat2.png",-1);
    //cout <<  m_rtHat << endl;
    if(abs(m_rtHat) < 20)
        imageRotate(m_Hat, ne, m_rtHat, false);
    else
    {
        return false;
    }

    //imageRotate(m_Hat, ne, m_rtHat, false);
    //imshow("m_Hat",m_Hat);

    Size dsize = Size(faceRect.width,faceRect.height);
    Mat hh ;
    resize(ne, hh,dsize);

    Mat newMat(frame, faceRect); 
    //Mat nn;
    //imageRotate(hh, nn, m_rtHat, false);
    cvAdd4cMat_q(newMat,hh,1.0); 

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

/*
int EXTfunction::prepareToProcessFaceDet()
{
	cascadeName = "./haarcascade_frontalface_alt.xml";
    nestedCascadeName = "./haarcascade_eye_tree_eyeglasses.xml";

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
}*/

// 加载人脸和左眼、右眼的检测器
void EXTfunction::initDetectors(CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2)
{
    // 载入人脸检测级联分类器-xml文件
    try 
    {   
        faceCascade.load(faceCascadeFilename);
    } 
    catch (cv::Exception &e) 
    { }
    if ( faceCascade.empty() ) 
    {
        cerr << "ERROR: 载入人脸检测级联分类器[" << faceCascadeFilename << "]失败!" << endl;
        exit(1);
    }
    cout << "载入人脸检测级联分类器[" << faceCascadeFilename << "]成功。" << endl;

    //// 载入眼睛检测级联分类器-xml文件 
    try {   
        eyeCascade1.load(eyeCascadeFilename1);
    } 
    catch (cv::Exception &e) 
    {}
    if ( eyeCascade1.empty() ) 
    {
        cerr << "ERROR:载入第一个眼睛检测级联分类器[" << eyeCascadeFilename1 << "]失败!" << endl;
       exit(1);
    }
    cout << "载入第一个眼睛检测级联分类器[" << eyeCascadeFilename1 << "]成功。" << endl;

    try {  
        eyeCascade2.load(eyeCascadeFilename2);
    } 
    catch (cv::Exception &e) 
    {}
    if ( eyeCascade2.empty() ) 
    {
        cerr << "ERROR:载入第二个眼睛检测级联分类器[" << eyeCascadeFilename2 << "]失败！" << endl;
        exit(1);
    }
    else
        cout << "载入第二个眼睛检测级联分类器[" << eyeCascadeFilename2 << "]成功。" << endl;
}

/*
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
        Size dsize = Size(r.width*m_eyeDecScale,r.height*m_eyeDecScale);
        resize( smallImgROI, smallImgROI, dsize);
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
            1.1, 2, 0
            //|CASCADE_FIND_BIGGEST_OBJECT
            //|CASCADE_DO_ROUGH_SEARCH
            //|CASCADE_DO_CANNY_PRUNING
            |CASCADE_SCALE_IMAGE,
            Size(30, 30) );
        if(nestedObjects.size() == 2)
        {
            for ( size_t j = 0; j < nestedObjects.size(); j++ )
        {
            Rect nr = nestedObjects[j];
                nr = Rect(nr.x/m_eyeDecScale,nr.y/m_eyeDecScale,nr.width/m_eyeDecScale,nr.height/m_eyeDecScale);
                center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
                center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
                radius = cvRound((nr.width + nr.height)*0.25*scale);
                circle( img, center, radius, color, 3, 8, 0 );
            }
        }
        
        faces[i] = cv::Rect(faces[i].x * m_Scale,faces[i].y *m_Scale,faces[i].width*m_Scale,faces[i].height*m_Scale);
    }

    t = (double)getTickCount() - t;
    //printf( "detection time = %g ms\n", t*1000/getTickFrequency());
    //imshow( "result", img );
} 
*/
bool EXTfunction::ProcessFaceDetect(cv::Mat* in)
{
	changeDetMat(in);
    //faces.clear();
    //检测到人脸并进行预处理（需要标准大小，对比度和亮度）
    double fx = 1 / m_Scale;
    Mat in2   =  in->clone();
    resize( in2, in2, Size(), fx, fx, INTER_LINEAR );

    Point leftEyeTemp,rightEyeTemp;
    Rect faceRectTemp;

    Mat preprocessedFace = getPreprocessedFace(in2, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRectTemp, &leftEyeTemp, &rightEyeTemp, &searchedLeftEye, &searchedRightEye);
    //faces.push_back(faceRect);
    gotFaceAndEyes = false;
    if (preprocessedFace.data)
        gotFaceAndEyes = true;

    if(5 >= faceRectTemp.width ||  5 >= faceRectTemp.height)
    {
        m_eyesStaticCount ++;
        m_faceStaticCount ++;
        return false;
    }else
    {
        faceRect = cv::Rect(faceRectTemp.x * m_Scale,faceRectTemp.y *m_Scale,faceRectTemp.width*m_Scale,faceRectTemp.height*m_Scale);
    }
    
    if(5 >= leftEyeTemp.x || 5 >= leftEyeTemp.y || 5 >=  rightEyeTemp.x  || 5 >= leftEyeTemp.y)
    {
        m_eyesStaticCount ++;
        return true;
    }else
    {
        leftEye = leftEyeTemp;
        rightEye  = rightEyeTemp;
        leftEye *=  m_Scale;
        rightEye *=   m_Scale;

        m_eyesWeight = Point(faceRect.width  *  EYE_IN_FACE_WIDTH_PER,faceRect.height * EYE_IN_FACE_HEIGHT_PER);
    }
    m_faceStaticCount =  0;
    m_eyesStaticCount  = 0;

    //Point  face_center ;
    // 在检测到的人脸周围绘制一个矩形
    /*
    if (faceRect.width > 0) 
    {
        //face_center =  Point(faceRect.x + faceRect.width/2, faceRect.y + faceRect.height/2);
        //rectangle(*in, faceRect, CV_RGB(255, 255, 0), 2, CV_AA);

        // 用蓝线画出眼睛的位置
        Scalar eyeColor = CV_RGB(0,255,255);
        if (leftEye.x >= 0 && rightEye.x >= 0) 
        {   
            //circle(*in, face_center, 6, eyeColor, 1, CV_AA);

            circle(*in, Point(faceRect.x + leftEye.x, faceRect.y + leftEye.y), 6, eyeColor, 1, CV_AA);

            circle(*in, Point(faceRect.x + rightEye.x, faceRect.y + rightEye.y), 6, eyeColor, 1, CV_AA);
        }
    }*/

    double ss = atan((double)(leftEye.y - rightEye.y)/(double)(leftEye.x-rightEye.x)) * 180/3.1415926;
    if(abs(ss) > 30  || abs(ss - m_rtHat)  > 30)
    {
        //
    }else
    {
        m_rtHat = ss;
    }
    //cout <<": " << m_rtHat << endl;

	//detectAndDraw( *m_dectedPreMat, cascade, nestedCascade, m_Scale , false );

	return true;
}

bool EXTfunction::ProcessEye(cv::Mat& in)
{
    if(5 >= leftEye.x || 5 >= leftEye.y || 5 >=  rightEye.x  || 5 >= leftEye.y)
    {
        return false;
    }

    if(m_faceStaticCount > FACE_STATIC_THE)
    {
        m_faceStaticCount = FACE_STATIC_THE;
        return false;
    }

    if(m_eyesStaticCount > EYE_STATIC_THE)
    {
        m_eyesStaticCount = EYE_STATIC_THE;
        return false;
    }

    // draw eyes
    Scalar eyeColor = CV_RGB(0,255,255);

    rectangle(in, faceRect, CV_RGB(255, 255, 0), 2, CV_AA);
    Rect  left = Rect(faceRect.x + leftEye.x - m_eyesWeight.x/2,faceRect.y + leftEye.y - m_eyesWeight.y/2,m_eyesWeight.x,m_eyesWeight.y);
    Rect  right = Rect(faceRect.x + rightEye.x - m_eyesWeight.x/2,faceRect.y + rightEye.y - m_eyesWeight.y/2,m_eyesWeight.x,m_eyesWeight.y);

    //rectangle(in, left, eyeColor, 1, CV_AA);

    //rectangle(in, right, eyeColor, 1, CV_AA);

    Mat  orgLeft =  in(left);
    Mat  orgRight =  in(right);

    //imshow("ResLeft", orgLeft);

    //imshow("ResRight", orgRight);
    /*
    IplImage* leftIpl;
    IplImage* rightIpl;
     */
    //IplImage leftIpl = IplImage(orgLeft);
    //IplImage rightIpl = IplImage(orgRight);

    MaxFrame(orgRight,0.3);

    MaxFrame(orgLeft,0.3);


    //MaxFrame(&rightIpl);
    //circle(*in, Point(faceRect.x + leftEye.x, faceRect.y + leftEye.y), 6, eyeColor, 1, CV_AA);
    //circle(*in, Point(faceRect.x + rightEye.x, faceRect.y + rightEye.y), 6, eyeColor, 1, CV_AA);

    //cout << leftEye << rightEye << endl;

    return true;
}

bool EXTfunction::ProcessFaceBeautification(cv::Mat& frame)
{
    if(!gotFaceAndEyes)
    {
        return false;
    }

    if(faceRect.width < 10 || faceRect.height < 10)
    {
        return false;
    }

	int KERNEL_SIZE = 15;  
	//cout <<": " << faceRect  << endl;
    int radius;
    Mat image_ori = frame(faceRect);
    Mat frameBfBil = image_ori.clone();
    for (int i = 1; i < KERNEL_SIZE; i = i + 2)  
    {  
        bilateralFilter(frameBfBil,image_ori,i,i*2,i/2);  
    } 
}