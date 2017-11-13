/*
* 这个文件用来显示Mat的结构
*/

#define USE_HIGHGUI        

#include "ImageUtils.h"

using namespace std;

//返回Mat中每个通道的位数，如: 8,16,32或64.
int getBitDepth(const cv::Mat M)
{
    switch (CV_MAT_DEPTH(M.type())) 
	{
        case CV_8U:
        case CV_8S:
            return 8;
        case CV_16U:
        case CV_16S:
            return 16;
        case CV_32S:
        case CV_32F:
            return 32;
        case CV_64F:
            return 64;
    }
    return -1;
}

//打印多通道数组的内容，方便调试(用"LOG()")
void printArray2D(const uchar *data, int cols, int rows, int channels, int depth_type, int step, int maxElements)
{
    
}

//打印图像或矩阵的内容(宽、高、通道数、每个通道位数)，方便调试代码(用"LOG()")
void printMat(const cv::Mat M, const char *label, int maxElements)
{
    
}

//// 打印图像或矩阵的信息(宽、高、通道数、每个通道位数)，方便调试代码(用"LOG()")
void printMatInfo(const cv::Mat M, const char *label)
{
    printMat(M, label, -1);
}
