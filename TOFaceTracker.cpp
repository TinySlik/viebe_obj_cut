#include "TOFaceTracker.h"

using namespace std;
using namespace cv;

pthread_mutex_t mutex_singleton_tofacetracker = PTHREAD_MUTEX_INITIALIZER;

//pthread safe sington
TOFaceTracker* TOFaceTracker::instance = NULL;
TOFaceTracker* TOFaceTracker::getInstance(){
    if (instance == NULL)
    {
        pthread_mutex_lock (&mutex_singleton_tofacetracker);
        if (instance == NULL)
        {
            instance = new TOFaceTracker();
        }
        pthread_mutex_unlock (&mutex_singleton_tofacetracker);
    }
    return instance;
}

// A detector for loading face and left eye and right eye
int TOFaceTracker::initDetectors(
const char *faceCascadeFilename,// LBP face detector.
const char *eyeCascadeFilename1,// Basic eye detector for open eyes only.
const char *eyeCascadeFilename2// Basic eye detector for open eyes if they might wear glasses.
)
{
    try 
    {   
        m_faceCascade.load(faceCascadeFilename);
    } 
    catch (cv::Exception &e) 
    { }
    if ( m_faceCascade.empty() ) 
    {
        cerr << "ERROR:[initDetectors] Cascade classifier carrying face detection[" << faceCascadeFilename << "]failed!" << endl;
        return -1;
    }
    cout << "[initDetectors] Cascade classifier carrying face detection[" << faceCascadeFilename << "]success!" << endl;
 
    try {   
        m_eyeCascade1.load(eyeCascadeFilename1);
    } 
    catch (cv::Exception &e) 
    {}
    if ( m_eyeCascade1.empty() ) 
    {
        cerr << "ERROR:[initDetectors] Load the first eye detection cascade classifier[" << eyeCascadeFilename1 << "]failed!" << endl;
       return -1;
    }
    cout << "[initDetectors] Load the first eye detection cascade classifier[" << eyeCascadeFilename1 << "]success." << endl;

    try {  
        m_eyeCascade2.load(eyeCascadeFilename2);
    } 
    catch (cv::Exception &e) 
    {}
    if ( m_eyeCascade2.empty() ) 
    {
        cerr << "ERROR:[initDetectors] Load the second eye detection cascade classifier[" << eyeCascadeFilename2 << "]failed!" << endl;
        return -1;
    }
    else
        cout << "[initDetectors] Load the second eye detection cascade classifier[" << eyeCascadeFilename2 << "]success." << endl;
    return 0;
}

TO_FACE_TRACKER_STATUS  TOFaceTracker::init(
        const char *faceCascadeFilename ,// LBP face detector.
        const char *eyeCascadeFilename1 ,// Basic eye detector for open eyes only.
        const char *eyeCascadeFilename2 ,// Basic eye detector for open eyes if they might wear glasses.
        unsigned long threadsWaitTimeInterval ,
        unsigned long faceDetectDormantWaitInterval ,
        unsigned short faceDetectCountEnableThreshold ,
        int updateStrategy
        )
{
    m_threadsWaitTimeInterval = threadsWaitTimeInterval;
    m_faceDetectDormantWaitInterval = faceDetectDormantWaitInterval;
    m_faceDetectCountEnableThreshold = faceDetectCountEnableThreshold;

    if(0 == updateStrategy)
    {
        m_enableSelfAdaptTimeCycle = true;
    }else if(updateStrategy > 0)
    {
        m_enableSelfAdaptTimeCycle = true;
        m_selfAdaptTimeCycleThreshold  = updateStrategy;
    }else if(updateStrategy < 0)
    {
        m_enableSelfAdaptTimeCycle = false;
        m_trackerNormalKeepTime  = 0-(updateStrategy);
    }

    if(STATUS_FACE_TRACKER_STOP != m_CurrentStatus)
    {
        cerr << "[TOFaceTracker init] reinit this class." << endl;
        m_CurrentStatus =  STATUS_FACE_TRACKER_ERRO;
        return STATUS_FACE_TRACKER_ERRO;
    }
    int retD = initDetectors(faceCascadeFilename,eyeCascadeFilename1,eyeCascadeFilename2);
    if (retD != 0)
    {
        cerr << "[TOFaceTracker init] initDetectors erro" << retD << endl;
        m_CurrentStatus =  STATUS_FACE_TRACKER_ERRO;
        return STATUS_FACE_TRACKER_ERRO;
    }

    m_userEnableSwitch = true;

    int retT = pthread_create(&faceAnalysisThreadID, NULL, TOFaceTracker::faceAnalysis, NULL);
    if (retT != 0)
    {
        cerr << "[TOFaceTracker init] face analysis pthread_create error: error_code=" << retT << endl;
        m_CurrentStatus =  STATUS_FACE_TRACKER_ERRO;
        return STATUS_FACE_TRACKER_ERRO;
    }

    retT = pthread_create(&KFCtracker1ThreadID, NULL, TOFaceTracker::kfcTracker, NULL);
    if (retT != 0)
    {
        cerr << "[TOFaceTracker init] KFC tracker1 pthread_create error: error_code=" << retT << endl;
        m_CurrentStatus =  STATUS_FACE_TRACKER_ERRO;
        return STATUS_FACE_TRACKER_ERRO;
    }

    retT = pthread_create(&KFCtracker2ThreadID, NULL, TOFaceTracker::kfcTracker, NULL);
    if (retT != 0)
    {
        cerr << "[TOFaceTracker init] KFC tracker2 pthread_create error: error_code=" << retT << endl;
        m_CurrentStatus =  STATUS_FACE_TRACKER_ERRO;
        return STATUS_FACE_TRACKER_ERRO;
    }

    m_isIntialed = true;
    return m_CurrentStatus;
}

TO_FACE_TRACKER_STATUS TOFaceTracker::pause()
{
    m_faceExistTag = false;
    m_userEnableSwitch = false;
    usleep(2000000);
    m_trackerCheckReadyTag = false;
    m_CurrentStatus = STATUS_FACE_TRACKER_STOP;
    m_isIntialed = false;
    while(m_FrameQueue.size() > 0)
    {
        frameQueuePop();
    }
    return m_CurrentStatus;
}

TO_FACE_TRACKER_STATUS TOFaceTracker::resume()
{
    m_CurrentStatus = STATUS_FACE_TRACKER_STOP;

    m_userEnableSwitch = true;
    m_isIntialed = true;
    return m_CurrentStatus;
}

TO_FACE_TRACKER_STATUS TOFaceTracker::sampling(const uchar* pBGRA,size_t width,size_t height,bool isDetectAlpha)
{
    if(!m_userEnableSwitch)
    {
#if _DEBUG_TO_FACE_TRACKER
        printf( "\n[TOFaceTracker sampling] in m_userEnableSwitch false status.\n");
#endif
        return m_CurrentStatus;
    }

    if(!m_isIntialed)
    {
        return m_CurrentStatus;
    }

    TO_QUEUE_FRAME frameData;
    
    uchar* dataIn = const_cast<uchar*>(pBGRA);
    Mat* frame = new Mat (height,width,CV_8UC3);
    frameData.data = frame;
    frameData.status = m_CurrentStatus;
    uchar* outData=frame->ptr<uchar>(0);
    if(isDetectAlpha)
    {
        for(int i=0;i<height ;i++)
        {
            for(int j=0;j<width;j++)
            {
                *outData++= *dataIn++;
                *outData++= *dataIn++;
                *outData++= *dataIn++;
                dataIn++;
            }
        }
    }else
    {
        for(int i=0;i<height ;i++)
        {
            for(int j=0;j<width;j++)
            {
                if(dataIn[3])
                {
                    *outData++= *dataIn++;
                    *outData++= *dataIn++;
                    *outData++= *dataIn++;
                    dataIn++;
                }else
                {
                    outData += 3;
                    dataIn += 4;
                }
            }
        }
    }
    
    frameData.timeStamp = getTickCount();
    m_FrameQueue.push(frameData);
    if(STATUS_FACE_TRACKER_STOP == m_CurrentStatus)
    {
        gotoNextStatus();
    }
    return m_CurrentStatus;
}

TO_FACE_TRACKER_STATUS  TOFaceTracker::frameQueuePop()
{
    if(m_FrameQueue.empty())
    {
        m_CurrentStatus =  STATUS_FACE_TRACKER_ERRO;
        cerr << "[TOFaceTracker pop]  queue empty "  << endl;
        return STATUS_FACE_TRACKER_ERRO;
    }
    delete  m_FrameQueue.front().data;
    m_FrameQueue.front().data = NULL;
    m_FrameQueue.pop();
}

TO_FACE_TRACKER_STATUS  TOFaceTracker::check(cv::Rect& face,unsigned int blockWaitTime,cv::Point* eyeLeft,cv::Point* eyeRight ,uchar* pBGRA)
{
    if(!m_userEnableSwitch)
    {
#if _DEBUG_TO_FACE_TRACKER
        printf( "\n[TOFaceTracker check] in m_userEnableSwitch false status.\n");
#endif
        face = Rect0;
        if(eyeLeft)
            *eyeLeft = Point0;
        if(eyeRight)
            *eyeRight = Point0;
        cout << "[TOFaceTracker check] pause , please use resume to continue." << endl;
        return m_CurrentStatus;
    }

    if(STATUS_FACE_TRACKER_INIT_FACE == m_CurrentStatus || STATUS_FACE_TRACKER_ERRO == m_CurrentStatus || STATUS_FACE_TRACKER_INIT_FACE== m_CurrentStatus || STATUS_FACE_TRACKER_INIT_TRACE== m_CurrentStatus)
    {
#if _DEBUG_TO_FACE_TRACKER
        printf("[TOFaceTracker check]please wait to check until the class init status in normal status");
        return  getRunStatus(true);
#else
        return m_CurrentStatus;
#endif
    }

    if(!instance -> m_faceExistTag)
    {
#if _DEBUG_TO_FACE_TRACKER
        printf("[TOFaceTracker check]no face found\n");
#endif
        face = Rect0;
        return m_CurrentStatus;
    }
    static unsigned long timeCounter ;
    if(STATUS_FACE_TRACKER_RUN_1_NORMAL == m_CurrentStatus || STATUS_FACE_TRACKER_RUN_1_FACE == m_CurrentStatus || STATUS_FACE_TRACKER_RUN_1_TRACE == m_CurrentStatus )
    {
        m_trackerCheckReadyTag = false;
        timeCounter = 0;
        while(!m_trackerCheckReadyTag)
        {
            usleep(1000);
            timeCounter++;
            if(timeCounter > blockWaitTime)
            {
                break;
            }
        }
        if(m_faceExistTag)
        {
            face = m_FaceRectTracker1;
        }else
        {
            face = Rect0;
        }
    }else if(STATUS_FACE_TRACKER_RUN_2_NORMAL == m_CurrentStatus || STATUS_FACE_TRACKER_RUN_2_FACE == m_CurrentStatus || STATUS_FACE_TRACKER_RUN_2_TRACE == m_CurrentStatus)
    {
        m_trackerCheckReadyTag = false;
        timeCounter = 0;
        while(!m_trackerCheckReadyTag)
        {
            usleep(1000);
            timeCounter++;
            if(timeCounter > blockWaitTime)
            {
                break;
            }
        }
        if(m_faceExistTag)
        {
            face = m_FaceRectTracker2;
        }else
        {
            face = Rect0;
        }
    }

    if(m_faceExistTag && eyeLeft && eyeRight)
    {
        ProcessEyesDetect(m_FrameQueue.back().data,face,* eyeLeft,* eyeRight);
    }

    if(pBGRA)
    {
        debugRes(pBGRA,m_FrameQueue.back().data,face ,eyeLeft,eyeRight);
    }

    return m_CurrentStatus;
}

void* TOFaceTracker::faceAnalysis(void* args)
{
    pthread_detach(pthread_self());
    while(true)
    {
        if(!instance ->m_userEnableSwitch)
        {
#if _DEBUG_TO_FACE_TRACKER
            printf( "\n[TOFaceTracker faceAnalysis] in m_userEnableSwitch false status.\n");
#endif
            usleep(2000000);
            continue;
        }
        TO_FACE_TRACKER_STATUS status =  instance ->getRunStatus(false);
        if(STATUS_FACE_TRACKER_ERRO == status)
        {
            break;
        }else if(STATUS_FACE_TRACKER_RUN_1_FACE  == status || STATUS_FACE_TRACKER_RUN_2_FACE  == status || STATUS_FACE_TRACKER_INIT_FACE  == status)
        {
            while(instance -> m_FrameQueue.size() > 1)
            {

                instance ->frameQueuePop();
            }
            cv::Rect res;
            if(instance -> ProcessFaceDetect(instance -> m_FrameQueue.front().data,res))
            {
                instance -> m_faceExistTag = true;
                instance -> m_FaceRect = res;
                instance -> m_noneDectCount = 0; 
                instance -> gotoNextStatus();
            }else
            {
                instance ->m_noneDectCount ++;
                if(instance ->m_noneDectCount > instance ->m_faceDetectCountEnableThreshold)
                {
                    instance ->m_faceExistTag = false;
                    instance ->m_noneDectCount = instance ->m_faceDetectCountEnableThreshold+1;
#if _DEBUG_TO_FACE_TRACKER
                    printf( "\n[TOFaceTracker] face detect start dormant time:%ld us.\n",instance ->m_faceDetectDormantWaitInterval);
#endif
                    usleep(instance ->m_faceDetectDormantWaitInterval);
                }else
                {
                    usleep(instance ->m_faceDetectWaitInterval);
                }
            }
            
        }else
        {
            usleep(instance ->m_threadsWaitTimeInterval);
            continue;
        }
    }
}

void* TOFaceTracker::kfcTracker(void* args)
{
    pthread_t sf = pthread_self();
    int kcfID = sf == instance -> KFCtracker1ThreadID ? 1 : 2;
    pthread_detach(sf);
    while(true)
    {
        if(!instance ->m_userEnableSwitch)
        {
#if _DEBUG_TO_FACE_TRACKER
            printf( "\n[TOFaceTracker kfcTracker] in m_userEnableSwitch false status.\n");
#endif
            usleep(2000000);
            continue;
        }
        TO_FACE_TRACKER_STATUS status =  instance ->getRunStatus(false);
        if(instance ->m_FrameQueue.size()  > 100)
        {
            cerr << "[TOFaceTracker err] m_FrameQueue.size()  > 100" << endl;
        } 
        if(STATUS_FACE_TRACKER_ERRO == status)
        {
            break;
        }else if(STATUS_FACE_TRACKER_STOP == status)
        {
            usleep(instance ->m_threadsWaitTimeInterval);
            continue;
        }else if(STATUS_FACE_TRACKER_INIT_FACE  == status)
        {
            usleep(instance ->m_threadsWaitTimeInterval);
            continue;
        }else if(STATUS_FACE_TRACKER_INIT_TRACE  == status)
        {
            unsigned long now = getTickCount();
            if(1 == kcfID)
            {
                unsigned long avgTimeInterval;
                while(instance -> m_FrameQueue.size() == 0)
                {
                    cerr << "[kfcTracker1] no date in queue!"  << endl;
                    usleep(100);
                }

                instance ->InitKCFTracker(instance ->m_Tracker1, instance -> m_FrameQueue.front().data,instance -> m_FaceRect);
                if(instance -> m_FrameQueue.size() > 1)
                {
                    avgTimeInterval = (instance -> m_FrameQueue.back().timeStamp - instance -> m_FrameQueue.front().timeStamp)/(instance -> m_FrameQueue.size() - 1);
                }else if(instance ->m_FrameQueue.size() == 1)
                {
                    avgTimeInterval = now - instance -> m_FrameQueue.front().timeStamp;
                }
                
                while(instance -> m_FrameQueue.size() > 0)
                {
                    bool waitToBreaK= false;
                    if(1 == instance -> m_FrameQueue.size())
                    {
                        waitToBreaK = true;
                    }
                    unsigned long custKFCTimeInterval = getTickCount();
                    instance ->ProcessKCFTracker(instance ->m_Tracker1, instance -> m_FrameQueue.front().data,instance -> m_FaceRectTracker1);
                    custKFCTimeInterval -= getTickCount();
                    unsigned int skipCount = 1;
                    //=======================================================strategy/=======================================================
                    if(avgTimeInterval - custKFCTimeInterval > 0)
                    {
                        //normal
                    }else
                    {
                        unsigned int tempNum ;
                        do
                        {
                            skipCount ++;
                            tempNum = avgTimeInterval/skipCount;
                        }while(custKFCTimeInterval - tempNum < 0);
                    }
                    if(waitToBreaK)
                    {
                        break;
                    }
                    //printf("[TOFaceTracker::kfcTracker1]size %d avgTimeInterval %lf,skipCount %d",instance ->m_FrameQueue.size(),((double)avgTimeInterval)*1000/getTickFrequency(),skipCount);

                    for (int i = 0; i < skipCount; ++i)
                    {
                        if(instance -> m_FrameQueue.size() > 1)
                        {
                            instance -> frameQueuePop();
                        }
                    }
                    //=======================================================/strategy=======================================================
                }
                instance ->gotoNextStatus();
            }
        }else if(STATUS_FACE_TRACKER_RUN_1_NORMAL  == status)
        {
            unsigned long startTime = getTickCount();
            if(1 == kcfID)
            {   
                static double distanceSum;
                distanceSum = 0;
                while(true)
                {
                    unsigned long now = getTickCount();
                    //=======================================================strategy/=======================================================
                    if(instance -> m_enableSelfAdaptTimeCycle)
                    {
                        static Rect rtNow =  instance -> m_FaceRectTracker1;
                        if(rtNow != instance ->m_FaceRectTracker1)
                        {
                            Point center1 = Point(rtNow.x + rtNow.width/2 , rtNow.y + rtNow.height/2);
                            Point center2 = Point(instance -> m_FaceRectTracker1.x + instance -> m_FaceRectTracker1.width/2 , instance -> m_FaceRectTracker1.y + instance -> m_FaceRectTracker1.height/2);
                            double distance = sqrtf(powf((center1.x - center2.x),2) + powf((center1.y - center2.y),2));
                            //cout << distance << endl;
                            
                            double areaValue1 = sqrtf(rtNow.width * rtNow.height);
                            double areaValue2 = sqrtf(instance -> m_FaceRectTracker1.width * instance -> m_FaceRectTracker1.height);
                            double areaValue = abs(areaValue1 - areaValue2)*10;

                            distanceSum += distance;
                            distanceSum += areaValue;
                            /*
                            cout <<"[STATUS_FACE_TRACKER_RUN_1_NORMAL] distanceSum :" << distanceSum << endl;
                            cout <<"[STATUS_FACE_TRACKER_RUN_1_NORMAL] areaValue :" << areaValue << endl;
                            cout <<"[STATUS_FACE_TRACKER_RUN_1_NORMAL] distance :" << distance << endl;
                            */
                           
                            if(instance -> m_FaceRectTracker1.x < 0 || 
                                instance -> m_FaceRectTracker1.y < 0 || 
                                (instance -> m_FaceRectTracker1.y + instance -> m_FaceRectTracker1.height) > (instance -> m_FrameQueue.back().data->rows) ||
                                (instance -> m_FaceRectTracker1.x + instance -> m_FaceRectTracker1.width) > (instance -> m_FrameQueue.back().data->cols)  )
                            {
                                break;
                            }
                            
                            if(distanceSum > instance ->m_selfAdaptTimeCycleThreshold)
                            {
                                break;
                            }
                            rtNow =  instance -> m_FaceRectTracker1;
                        }
                    }
                    else
                    {
                        if((double)(now - startTime)*1000/getTickFrequency() > instance -> m_trackerNormalKeepTime )
                        {
                            break;
                        }
                    }
                    
                    //=======================================================/strategy=======================================================
                    while(instance ->m_trackerCheckReadyTag)
                    {
                        usleep(500);
                    }

                    if(instance -> m_FrameQueue.size() == 0)
                    {
                        cerr  << "[STATUS_FACE_TRACKER_RUN_1_NORMAL]  size  = 0" << endl;
                        instance ->m_CurrentStatus = STATUS_FACE_TRACKER_ERRO;
                        break;
                    }
                    while(instance -> m_FrameQueue.size() > 1)
                    {
                        instance ->frameQueuePop();
                    }
                    //printf("\n[TOFaceTracker::kfcTracker1 STATUS_FACE_TRACKER_RUN_1_NORMAL ]time delay: %lf \n",(double)(getTickCount() - instance -> m_FrameQueue.front().timeStamp)*1000/getTickFrequency());
                    instance ->ProcessKCFTracker(instance ->m_Tracker1, instance -> m_FrameQueue.front().data,instance -> m_FaceRectTracker1);

                    instance ->m_trackerCheckReadyTag   = true;
                }
                instance ->gotoNextStatus();
            }
        }else if(STATUS_FACE_TRACKER_RUN_1_FACE  == status)
        {
            if(1 == kcfID)
            {   
                if(!instance -> m_faceExistTag)
                {
                    usleep(instance ->m_threadsWaitTimeInterval);
                    continue;
                }
                unsigned long now = getTickCount();
                while(instance ->m_trackerCheckReadyTag)
                {
                    usleep(500);
                }

                if(instance -> m_FrameQueue.size() == 0)
                {
                    cerr  << "[STATUS_FACE_TRACKER_RUN_1_FACE]  size  = 0" << endl;
                    instance ->m_CurrentStatus = STATUS_FACE_TRACKER_ERRO;
                    return NULL;
                }
                //printf("\n[TOFaceTracker::kfcTracker1 STATUS_FACE_TRACKER_RUN_1_NORMAL ]time delay: %lf \n",(double)(getTickCount() - instance -> m_FrameQueue.front().timeStamp)*1000/getTickFrequency());
                instance ->ProcessKCFTracker(instance ->m_Tracker1, instance -> m_FrameQueue.back().data,instance -> m_FaceRectTracker1);

                instance ->m_trackerCheckReadyTag   = true;
            }
        }else if(STATUS_FACE_TRACKER_RUN_1_TRACE  == status)
        {
            unsigned long now = getTickCount();

            if(1 == kcfID)
            {   
                while(instance ->m_trackerCheckReadyTag)
                {
                    usleep(500);
                }

                if(instance -> m_FrameQueue.size() == 0)
                {
                    cerr  << "[STATUS_FACE_TRACKER_RUN_1_FACE]  size  = 0" << endl;
                    instance ->m_CurrentStatus = STATUS_FACE_TRACKER_ERRO;
                    break;
                }
                //printf("\n[TOFaceTracker::kfcTracker1 STATUS_FACE_TRACKER_RUN_1_NORMAL ]time delay: %lf \n",(double)(getTickCount() - instance -> m_FrameQueue.front().timeStamp)*1000/getTickFrequency());
                instance ->ProcessKCFTracker(instance ->m_Tracker1, instance -> m_FrameQueue.back().data,instance -> m_FaceRectTracker1);

                instance ->m_trackerCheckReadyTag   = true;
            }
            
            if(2 == kcfID)
            {
                unsigned long avgTimeInterval;
                while(instance -> m_FrameQueue.size()  == 0)
                {
                    cerr << "[kfcTracker1] no date in queue!\n"  << endl;
                    usleep(100);
                }

                instance ->InitKCFTracker(instance ->m_Tracker2, instance -> m_FrameQueue.front().data,instance -> m_FaceRect);

                if(instance -> m_FrameQueue.size() > 1)
                {
                    avgTimeInterval = (instance -> m_FrameQueue.back().timeStamp - instance -> m_FrameQueue.front().timeStamp)/(instance -> m_FrameQueue.size() - 1);
                }else if(instance ->m_FrameQueue.size() == 1)
                {
                    avgTimeInterval = now - instance -> m_FrameQueue.front().timeStamp;
                }
                
                while(instance -> m_FrameQueue.size() > 0)
                {
                    bool waitToBreaK= false;
                    if(1 == instance -> m_FrameQueue.size())
                    {
                        waitToBreaK = true;;
                    }
                    unsigned long custKFCTimeInterval = getTickCount();
                    instance ->ProcessKCFTracker(instance ->m_Tracker2, instance -> m_FrameQueue.front().data,instance -> m_FaceRectTracker2);
                    custKFCTimeInterval -= getTickCount();
                    unsigned int skipCount = 1;
                    //=======================================================strategy/=======================================================
                    if(avgTimeInterval - custKFCTimeInterval > 0)
                    {
                        //normal
                    }else
                    {
                        unsigned int tempNum ;
                        do
                        {
                            skipCount ++;
                            tempNum = avgTimeInterval/skipCount;
                        }while(custKFCTimeInterval - tempNum < 0);
                    }
                    if(waitToBreaK)
                    {
                        break;
                    }
                    //printf("[TOFaceTracker::kfcTracker2]size %d avgTimeInterval %lf,skipCount %d",instance ->m_FrameQueue.size(),((double)avgTimeInterval)*1000/getTickFrequency(),skipCount);

                    for (int i = 0; i < skipCount; ++i)
                    {
                        if(instance -> m_FrameQueue.size() > 1)
                        {
                            instance -> frameQueuePop();
                        }
                    }
                    //=======================================================/strategy=======================================================
                }
                instance ->gotoNextStatus();
            }
        }else if(STATUS_FACE_TRACKER_RUN_2_NORMAL  == status)
        {
            unsigned long startTime = getTickCount();
            if(2 == kcfID)
            {   
                static double distanceSum;
                distanceSum = 0;
                while(true)
                {
                    unsigned long now = getTickCount();
                    //=======================================================strategy/=======================================================
                    if(instance -> m_enableSelfAdaptTimeCycle)
                    {
                        static Rect rtNow =  instance -> m_FaceRectTracker2;
                        if(rtNow != instance ->m_FaceRectTracker2)
                        {
                            Point center1 = Point(rtNow.x + rtNow.width/2 , rtNow.y + rtNow.height/2);
                            Point center2 = Point(instance -> m_FaceRectTracker2.x + instance -> m_FaceRectTracker2.width/2 , instance -> m_FaceRectTracker2.y + instance -> m_FaceRectTracker2.height/2);
                            double distance = sqrtf(powf((center1.x - center2.x),2) + powf((center1.y - center2.y),2));

                            double areaValue1 = sqrtf(rtNow.width * rtNow.height);
                            double areaValue2 = sqrtf(instance -> m_FaceRectTracker2.width * instance -> m_FaceRectTracker2.height);
                            double areaValue = abs(areaValue1 - areaValue2);
                            distanceSum += distance;
                            distanceSum += areaValue;
                            /*
                            cout <<"[STATUS_FACE_TRACKER_RUN_2_NORMAL] distanceSum :" << distanceSum << endl;
                            cout <<"[STATUS_FACE_TRACKER_RUN_2_NORMAL] areaValue :" << areaValue << endl;
                            cout <<"[STATUS_FACE_TRACKER_RUN_2_NORMAL] distance :" << distance << endl;
                            */
                           
                            if(instance -> m_FaceRectTracker2.x < 0 || 
                                instance -> m_FaceRectTracker2.y < 0 || 
                                (instance -> m_FaceRectTracker2.y + instance -> m_FaceRectTracker2.height) > (instance -> m_FrameQueue.back().data->rows) ||
                                (instance -> m_FaceRectTracker2.x + instance -> m_FaceRectTracker2.width) > (instance -> m_FrameQueue.back().data->cols)  )
                            {
                                break;
                            }

                            if(distanceSum > instance ->m_selfAdaptTimeCycleThreshold)
                            {
                                break;
                            }
                            rtNow =  instance -> m_FaceRectTracker2;
                        }
                    }
                    else
                    {
                        if((double)(now - startTime)*1000/getTickFrequency() > instance -> m_trackerNormalKeepTime )
                        {
                            break;
                        }
                    }
                    
                    //=======================================================/strategy=======================================================
                    while(instance ->m_trackerCheckReadyTag)
                    {
                        usleep(500);
                    }

                    if(instance -> m_FrameQueue.size() == 0)
                    {
                        cerr  << "[STATUS_FACE_TRACKER_RUN_1_NORMAL]  size  = 0" << endl;
                        instance ->m_CurrentStatus = STATUS_FACE_TRACKER_ERRO;
                        break;
                    }
                    while(instance -> m_FrameQueue.size() > 1)
                    {
                        instance ->frameQueuePop();
                    }
                    //printf("\n[TOFaceTracker::kfcTracker1 STATUS_FACE_TRACKER_RUN_1_NORMAL ]time delay: %lf \n",(double)(getTickCount() - instance -> m_FrameQueue.front().timeStamp)*1000/getTickFrequency());
                    instance ->ProcessKCFTracker(instance ->m_Tracker2, instance -> m_FrameQueue.front().data,instance -> m_FaceRectTracker2);

                    instance ->m_trackerCheckReadyTag   = true;
                }
                instance ->gotoNextStatus();
            }
        }else if(STATUS_FACE_TRACKER_RUN_2_FACE  == status)
        {
            if(2 == kcfID)
            {   
                if(!instance -> m_faceExistTag)
                {
                    usleep(instance ->m_threadsWaitTimeInterval);
                    continue;
                }
                unsigned long now = getTickCount();
                while(instance ->m_trackerCheckReadyTag)
                {
                    usleep(500);
                }

                if(instance -> m_FrameQueue.size() == 0)
                {
                    cerr  << "[STATUS_FACE_TRACKER_RUN_1_FACE]  size  = 0" << endl;
                    instance ->m_CurrentStatus = STATUS_FACE_TRACKER_ERRO;
                    return NULL;
                }
                //printf("\n[TOFaceTracker::kfcTracker1 STATUS_FACE_TRACKER_RUN_1_NORMAL ]time delay: %lf \n",(double)(getTickCount() - instance -> m_FrameQueue.front().timeStamp)*1000/getTickFrequency());
                instance ->ProcessKCFTracker(instance ->m_Tracker2, instance -> m_FrameQueue.back().data,instance -> m_FaceRectTracker2);

                instance ->m_trackerCheckReadyTag   = true;
            }
        }else if(STATUS_FACE_TRACKER_RUN_2_TRACE  == status)
        {
            unsigned long now = getTickCount();

            if(2 == kcfID)
            {   
                while(instance ->m_trackerCheckReadyTag)
                {
                    usleep(500);
                }

                if(instance -> m_FrameQueue.size() == 0)
                {
                    cerr  << "[STATUS_FACE_TRACKER_RUN_1_FACE]  size  = 0" << endl;
                    instance ->m_CurrentStatus = STATUS_FACE_TRACKER_ERRO;
                    break;
                }
                //printf("\n[TOFaceTracker::kfcTracker1 STATUS_FACE_TRACKER_RUN_1_NORMAL ]time delay: %lf \n",(double)(getTickCount() - instance -> m_FrameQueue.front().timeStamp)*1000/getTickFrequency());
                instance ->ProcessKCFTracker(instance ->m_Tracker2, instance -> m_FrameQueue.back().data,instance -> m_FaceRectTracker2);

                instance ->m_trackerCheckReadyTag   = true;
            }
            
            if(1 == kcfID)
            {
                unsigned long avgTimeInterval;
                while( instance -> m_FrameQueue.size()  == 0)
                {
                    cerr << "[kfcTracker1] no date in queue!\n"  << endl;
                    usleep(100);
                }

                instance ->InitKCFTracker(instance ->m_Tracker1, instance -> m_FrameQueue.front().data,instance -> m_FaceRect);

                if(instance -> m_FrameQueue.size() > 1)
                {
                    avgTimeInterval = (instance -> m_FrameQueue.back().timeStamp - instance -> m_FrameQueue.front().timeStamp)/(instance -> m_FrameQueue.size() - 1);
                }else if(instance ->m_FrameQueue.size() == 1)
                {
                    avgTimeInterval = now - instance -> m_FrameQueue.front().timeStamp;
                }
                
                while(instance -> m_FrameQueue.size() > 0)
                {
                    bool waitToBreaK= false;
                    if(1 == instance -> m_FrameQueue.size())
                    {
                        waitToBreaK = true;;
                    }
                    unsigned long custKFCTimeInterval = getTickCount();
                    instance ->ProcessKCFTracker(instance ->m_Tracker1, instance -> m_FrameQueue.front().data,instance -> m_FaceRectTracker1);
                    custKFCTimeInterval -= getTickCount();
                    unsigned int skipCount = 1;
                    //=======================================================strategy/=======================================================
                    if(avgTimeInterval - custKFCTimeInterval > 0)
                    {
                        //normal
                    }else
                    {
                        unsigned int tempNum ;
                        do
                        {
                            skipCount ++;
                            tempNum = avgTimeInterval/skipCount;
                        }while(custKFCTimeInterval - tempNum < 0);
                    }
                    if(waitToBreaK)
                    {
                        break;
                    }
                    //printf("[TOFaceTracker::kfcTracker2]size %d avgTimeInterval %lf,skipCount %d",instance ->m_FrameQueue.size(),((double)avgTimeInterval)*1000/getTickFrequency(),skipCount);

                    for (int i = 0; i < skipCount; ++i)
                    {
                        if(instance -> m_FrameQueue.size() > 1)
                        {
                            instance -> frameQueuePop();
                        }
                    }
                    //=======================================================/strategy=======================================================
                }
                instance ->gotoNextStatus();
            }
        }else
        {
            break;
        }
    }
}

TO_FACE_TRACKER_STATUS TOFaceTracker::gotoNextStatus()
{
    if(STATUS_FACE_TRACKER_RUN_2_TRACE == m_CurrentStatus)
    {
        m_CurrentStatus = STATUS_FACE_TRACKER_RUN_1_NORMAL;
    }
    else if( STATUS_FACE_TRACKER_ERRO  == m_CurrentStatus)
    {
        cout << "[TOFaceTracker] in erro status , anylisize and restart."  << endl;
    }
    else
    {
        m_CurrentStatus = (TO_FACE_TRACKER_STATUS)((int)m_CurrentStatus + 1);
    }

    return m_CurrentStatus;
}

TO_FACE_TRACKER_STATUS TOFaceTracker::getRunStatus(bool isNeedToPrint )
{
    if(isNeedToPrint)
    {
        printf("\n");
        switch(m_CurrentStatus)
        {
        case STATUS_FACE_TRACKER_STOP:
            printf("[TOFaceTracker] current status: STATUS_FACE_TRACKER_STOP");
            break;
        case STATUS_FACE_TRACKER_INIT_FACE:
            printf("[TOFaceTracker] current status:STATUS_FACE_TRACKER_INIT_FACE");
            break;
        case STATUS_FACE_TRACKER_INIT_TRACE:
            printf("[TOFaceTracker] current status:STATUS_FACE_TRACKER_INIT_TRACE");
            break;
        case STATUS_FACE_TRACKER_RUN_1_NORMAL:
            printf("[TOFaceTracker] current status:STATUS_FACE_TRACKER_RUN_1_NORMAL");
            break;
        case STATUS_FACE_TRACKER_RUN_1_FACE:
            printf("[TOFaceTracker] current status:STATUS_FACE_TRACKER_RUN_1_FACE");
            break;
        case STATUS_FACE_TRACKER_RUN_1_TRACE:
            printf("[TOFaceTracker] current status:STATUS_FACE_TRACKER_RUN_1_TRACE");
            break;
        case STATUS_FACE_TRACKER_RUN_2_NORMAL:
            printf("[TOFaceTracker] current status:STATUS_FACE_TRACKER_RUN_2_NORMAL");
            break;
        case STATUS_FACE_TRACKER_RUN_2_FACE:
            printf("[TOFaceTracker] current status:STATUS_FACE_TRACKER_RUN_2_FACE");
            break;
        case STATUS_FACE_TRACKER_RUN_2_TRACE:
            printf("[TOFaceTracker] current status:STATUS_FACE_TRACKER_RUN_2_TRACE");
            break;
        }
    }
    return m_CurrentStatus;
}

bool TOFaceTracker::ProcessFaceDetect(cv::Mat* in,cv::Rect& res ,float scale, size_t faceDoubleRadious,const bool preprocessLeftAndRightSeparately)
{
#if _DEBUG_TO_FACE_TRACKER
    double times = (double)getTickCount();
#endif
    if (NULL == in)
    {
        return false;
    }

    if(scale < 0.01  || scale > 2.0)
    {
        cout << "[ProcessFaceDetect] scale should  be  in (0.01 ~  2.0)." << endl;
        return false;
    }
    cv::Mat* srcData = in;
    if(scale < 0.999f || scale > 1.001f)
    {
        cv::Mat srcResize;
        srcData = &srcResize;
        cv::resize(*in, srcResize, cv::Size(scale*in->cols, scale*in->rows), (0, 0), (0, 0), cv::INTER_LINEAR);
    }

    Rect faceRectTemp;
    //cout  << "hello:" << srcData->rows  << srcData->cols << endl;
    if(getPreprocessedFaceOnly(*srcData, m_faceCascade, faceRectTemp))
    {
        res = cv::Rect(faceRectTemp.x /scale,faceRectTemp.y /scale, faceRectTemp.width/scale, faceRectTemp.height/scale);
        return true;
    }

#if _DEBUG_TO_FACE_TRACKER
    times = (double)getTickCount() - times;
    printf( "\n[TOFaceTracker][TIME]ProcessFaceDetect times cust frame time  %lf \n",times*1000/getTickFrequency());
#endif
    return false;
}

bool TOFaceTracker::ProcessEyesDetect(cv::Mat* in,const cv::Rect& face,cv::Point& eyeLeft,cv::Point& eyeRight)
{
#if _DEBUG_TO_FACE_TRACKER
    double times = (double)getTickCount();
#endif
    Mat faceImg = (*in)(face);    // 获得检测到的人脸图像
    Mat gray;
    if (faceImg.channels() == 3) {
        cvtColor(faceImg, gray, CV_BGR2GRAY);
    }
    else if (faceImg.channels() == 4) {
        cvtColor(faceImg, gray, CV_BGRA2GRAY);
    }
    else {
        gray = faceImg;
    }
    detectBothEyes(gray, m_eyeCascade1, m_eyeCascade2, eyeLeft, eyeRight);
#if _DEBUG_TO_FACE_TRACKER
    times = (double)getTickCount() - times;
    printf( "\n[TOFaceTracker][TIME]ProcessEyesDetect times cust frame time  %lf \n",times*1000/getTickFrequency());
#endif
}

bool TOFaceTracker::ProcessKCFTracker(KCFTracker& kcftracker,cv::Mat* in,cv::Rect& rect)
{
#if _DEBUG_TO_FACE_TRACKER
    double times = (double)getTickCount();
#endif
    rect = kcftracker.update(*in);
#if _DEBUG_TO_FACE_TRACKER
    times = (double)getTickCount() - times;
    printf( "\n[TOFaceTracker][TIME]ProcessKCFTracker times cust frame time  %lf \n",times*1000/getTickFrequency());
#endif
}

bool TOFaceTracker::InitKCFTracker(KCFTracker& kcftracker,cv::Mat* in,const cv::Rect& rect )
{
#if _DEBUG_TO_FACE_TRACKER
    double times = (double)getTickCount();
#endif
    kcftracker.init( rect , *in);
#if _DEBUG_TO_FACE_TRACKER
    times = (double)getTickCount() - times;
    printf( "\n[TOFaceTracker][TIME]InitKCFTracker times cust frame time  %lf \n",times*1000/getTickFrequency());
#endif
}

bool TOFaceTracker::debugRes(uchar* pBGRA,cv::Mat* in,cv::Rect faceRect ,cv::Point* leftEye,cv::Point* rightEye)
{
    if (m_faceExistTag && faceRect.width * faceRect.height > 0) 
    {
        //face_center =  Point(faceRect.x + faceRect.width/2, faceRect.y + faceRect.height/2);
        rectangle(*in, faceRect, CV_RGB(255, 255, 0), 2, CV_AA);
        Scalar eyeColor = CV_RGB(0,255,255);
        if(leftEye && rightEye)
        {
            if (leftEye->x >= 0 && rightEye->x >= 0) 
            {   
                //circle(*in, face_center, 6, eyeColor, 1, CV_AA);
                circle(*in, Point(faceRect.x + leftEye->x, faceRect.y + leftEye->y), 6, eyeColor, 1, CV_AA);
                circle(*in, Point(faceRect.x + rightEye->x, faceRect.y + rightEye->y), 6, eyeColor, 1, CV_AA);
            }
        }
    }

    if(m_faceExistTag)
    {
        rectangle(*in, m_FaceRect, CV_RGB(255,0, 0), 2, CV_AA);
        uchar* outData=in->ptr<uchar>(0);
        for(int i=0;i< in->rows;i++)
        {
            for(int j=0;j< in->cols;j++)
            {
                *pBGRA++= *outData++;
                *pBGRA++= *outData++;
                *pBGRA++= *outData++;
                pBGRA++;
            }
        }
    }
}