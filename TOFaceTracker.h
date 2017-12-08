#ifndef TO_FACE_TRACKER_H
#define TO_FACE_TRACKER_H

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <queue>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "kcftracker.hpp"
#include "preprocessFace.h"

#define FACE_CASCADE_FILENAME_DEFAULT "../cv_resource/haarcascade_frontalface_alt.xml" //path*
#define EYE1_CASCADE_FILENAME_DEFAULT "../cv_resource/haarcascade_eye.xml"  //path*
#define EYE2_CASCADE_FILENAME_DEFAULT "../cv_resource/haarcascade_eye_tree_eyeglasses.xml" //path*

//#define EYE1_CASCADE_FILENAME_DEFAULT "../cv_resource/haarcascade_lefteye_2splits.xml"  //path*
//#define EYE2_CASCADE_FILENAME_DEFAULT "../cv_resource/haarcascade_righteye_2splits.xml" //path*

#define Rect0 cv::Rect(0,0,0,0)
#define Point0 cv::Point(0,0)

#define _DEBUG_TO_FACE_TRACKER false

#define DETECT_ALPHA false

#define THREAS_WAIT_TIME_INTERVAL  20000 //us*

#define FACE_DETECT_WAIT_INTERVAL 200000 //us

#define FACE_DETECT_DORMANT_WAIT_INTERVAL 2000000 //us*

#define FACE_DETECT_COUNT_ENABLE_THRESHOLD 5 // count*

#define FACE_OUT_PUT_WAIT_TIME_DEFAULT 1000  

#define NORMAL_TIME_VINVERTAL 3000 //ms*

#define SELF_ADAPT_TIME_CYCLE_THRESHOLD 500 // weight*

// kcf tracker config
#define HOG false
#define FIXEDWINDOW false
#define MULTISCALE true
#define SILENT true
#define LAB false

typedef enum {
    STATUS_FACE_TRACKER_STOP = 0,
    STATUS_FACE_TRACKER_INIT_FACE ,
    STATUS_FACE_TRACKER_INIT_TRACE ,
    STATUS_FACE_TRACKER_RUN_1_NORMAL ,
    STATUS_FACE_TRACKER_RUN_1_FACE ,
    STATUS_FACE_TRACKER_RUN_1_TRACE ,
    STATUS_FACE_TRACKER_RUN_2_NORMAL ,
    STATUS_FACE_TRACKER_RUN_2_FACE ,
    STATUS_FACE_TRACKER_RUN_2_TRACE ,
    STATUS_FACE_TRACKER_ERRO ,
} TO_FACE_TRACKER_STATUS;

typedef struct {
    cv::Mat* data;
    unsigned long timeStamp;
    TO_FACE_TRACKER_STATUS status;
}TO_QUEUE_FRAME;

class TOFaceTracker
{

public:
    virtual ~TOFaceTracker(){};
    static TOFaceTracker* getInstance();

private:
    TOFaceTracker():
    m_isIntialed(false),
    m_faceExistTag(false),
    m_CurrentStatus(STATUS_FACE_TRACKER_STOP),
    m_FaceRect(0,0,0,0),
    m_Tracker1(HOG, FIXEDWINDOW, MULTISCALE, LAB),
    m_Tracker2(HOG, FIXEDWINDOW, MULTISCALE, LAB),
    m_trackerCheckReadyTag(false),
    m_trackerNormalKeepTime(NORMAL_TIME_VINVERTAL),
    m_faceDetectWaitInterval(FACE_DETECT_WAIT_INTERVAL),
    m_faceDetectDormantWaitInterval(FACE_DETECT_DORMANT_WAIT_INTERVAL),
    m_faceDetectCountEnableThreshold(FACE_DETECT_COUNT_ENABLE_THRESHOLD),
    m_threadsWaitTimeInterval(THREAS_WAIT_TIME_INTERVAL),
    m_noneDectCount(0),
    m_enableSelfAdaptTimeCycle(true),
    m_selfAdaptTimeCycleThreshold(SELF_ADAPT_TIME_CYCLE_THRESHOLD),
    m_userEnableSwitch(false)
    {};
    inline TOFaceTracker(const TOFaceTracker&){};
    inline TOFaceTracker& operator=(const TOFaceTracker&){};
    static TOFaceTracker* instance;

public:
    TO_FACE_TRACKER_STATUS init(
        // these 3 Cascade  path is the main detect tool of face detect. 
        const char *faceCascadeFilename = FACE_CASCADE_FILENAME_DEFAULT,
        const char *eyeCascadeFilename1 = EYE1_CASCADE_FILENAME_DEFAULT, 
        const char *eyeCascadeFilename2 = EYE2_CASCADE_FILENAME_DEFAULT,
        // 线程的休眠间隔最小单位，就是说隔多久对整体的状态做一次反应，不可设置太大 单位：us
        unsigned long threadsWaitTimeInterval = THREAS_WAIT_TIME_INTERVAL,
        // 脸部检测线程休眠判定时间 单位：us
        unsigned long faceDetectDormantWaitInterval = FACE_DETECT_DORMANT_WAIT_INTERVAL,
        // 脸部检测休眠检测判定需要的检测失败次数 
        unsigned short faceDetectCountEnableThreshold = FACE_DETECT_COUNT_ENABLE_THRESHOLD,
        // 自适应的主检测更新策略 
        // 填写正数时视为临界判定的累加权重阀值
        // 填写负数时视为使用固定时间更新的策略，单位：ms
        int updateStrategy   =  0
        );
    TO_FACE_TRACKER_STATUS sampling(
        const uchar* pBGRA,
        size_t width,
        size_t height ,
        bool isDetectAlpha = DETECT_ALPHA);
    TO_FACE_TRACKER_STATUS check(
        cv::Rect& face,
        unsigned int blockWaitTime = FACE_OUT_PUT_WAIT_TIME_DEFAULT , // ms
        cv::Point* eyeLeft =NULL,
        cv::Point* eyeRight = NULL,
        uchar* pBGRA = NULL);
    TO_FACE_TRACKER_STATUS getRunStatus(
        bool isNeedToPrint = _DEBUG_TO_FACE_TRACKER);

    TO_FACE_TRACKER_STATUS pause();
    TO_FACE_TRACKER_STATUS resume();

    bool isFaceExist(){return m_faceExistTag;};

private:
    TO_FACE_TRACKER_STATUS gotoNextStatus();
    TO_FACE_TRACKER_STATUS frameQueuePop();

    int initDetectors(
        const char *faceCascadeFilename ,// LBP face detector.
        const char *eyeCascadeFilename1 ,// Basic eye detector for open eyes only.
        const char *eyeCascadeFilename2  // Basic eye detector for open eyes if they might wear glasses.
    );

    bool ProcessFaceDetect(cv::Mat* in,cv::Rect& res,float scale = 1.0f , size_t faceDoubleRadious = 70,const bool preprocessLeftAndRightSeparately  = true);
    bool ProcessEyesDetect(cv::Mat* in,const cv::Rect& face,cv::Point& eyeLeft,cv::Point& eyeRight);

    bool ProcessKCFTracker(KCFTracker& kcftracker, cv::Mat* in,cv::Rect& rect);
    bool InitKCFTracker(KCFTracker& kcftracker,cv::Mat* in,const cv::Rect& rect);

    pthread_t  faceAnalysisThreadID;
    pthread_t  KFCtracker1ThreadID;
    pthread_t  KFCtracker2ThreadID;

    static void* faceAnalysis(void* args);
    static void* kfcTracker(void* args);

    bool debugRes(uchar* pBGRA,cv::Mat* in,cv::Rect face ,cv::Point* eyel,cv::Point* eyeR);

    cv::CascadeClassifier m_faceCascade;
    cv::CascadeClassifier m_eyeCascade1;
    cv::CascadeClassifier m_eyeCascade2;

    bool m_isIntialed;

    std::queue<TO_QUEUE_FRAME> m_FrameQueue;

    TO_FACE_TRACKER_STATUS m_CurrentStatus;

    cv::Rect m_FaceRect;  
    cv::Rect m_FaceRectTracker1;
    cv::Rect m_FaceRectTracker2;

    KCFTracker m_Tracker1;
    KCFTracker m_Tracker2;

    bool m_userEnableSwitch;

    bool m_trackerCheckReadyTag;
    bool m_faceExistTag;
    bool m_enableSelfAdaptTimeCycle;

    unsigned int m_selfAdaptTimeCycleThreshold;

    unsigned long m_trackerNormalKeepTime;//ms
    unsigned long m_threadsWaitTimeInterval;
    unsigned long m_faceDetectWaitInterval;
    unsigned long m_faceDetectDormantWaitInterval;
    unsigned short m_faceDetectCountEnableThreshold;
    unsigned short m_noneDectCount;
};

#endif //TO_FACE_TRACKER_H