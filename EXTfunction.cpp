#include "EXTfunction.h"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "vibe-background-sequential.h"

using namespace std;
using namespace cv;

EXTfunction::EXTfunction()
{
	//to do
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
	Size dsize = Size(in->cols/2,in->rows/2);
	Mat myMat= imread("frameBox3.png",-1); 

    imageRotate(myMat, myMat, 30, false);
	resize(myMat, myMat,dsize);

	Mat img1_t1(*in, cvRect(0, 0, myMat.cols, myMat.rows));  
	cvAdd4cMat_q(img1_t1,myMat,1.0);  
	//imshow("Resoult2", img1_t1);
	
	//*in = *in + myMat;
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