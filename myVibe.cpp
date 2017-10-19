#include "myVibe.h"

myVibe::myVibe()
{
    model = NULL;
}
myVibe::~myVibe()
{
    libvibeModel_Sequential_Free(model);
}
bool myVibe::ProcessVideo(cv::Mat* inputFrame, cv::Mat* segmentationMap, long frameNumber)
{
    if ((frameNumber % 100) == 0) { cout << "Frame number = " << frameNumber << endl; }

    if (frameNumber == 1) {
      *segmentationMap = Mat(inputFrame->rows, inputFrame->cols, CV_8UC1);
      model = (vibeModel_Sequential_t*)libvibeModel_Sequential_New();
      libvibeModel_Sequential_AllocInit_8u_C3R(model, inputFrame->data, inputFrame->cols, inputFrame->rows);
    }

    /* ViBe: Segmentation and updating. */
    libvibeModel_Sequential_Segmentation_8u_C3R(model, inputFrame->data, segmentationMap->data);
    libvibeModel_Sequential_Update_8u_C3R(model, inputFrame->data, segmentationMap->data);

    /* Post-processes the segmentation map. This step is not compulsory.
       Note that we strongly recommend to use post-processing filters, as they
       always smooth the segmentation map. For example, the post-processing filter
       used for the Change Detection dataset (see http://www.changedetection.net/ )
       is a 5x5 median filter. */
    medianBlur(*segmentationMap, *segmentationMap, 3); /* 3x3 median filtering */
    return true;
}
