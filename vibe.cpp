#include "vibe.h"
using namespace std;
#define NUMBER_OF_HISTORY_IMAGES 2

static inline int abs_uint(const int i)
{
    return (i >= 0) ? i : -i;
}

static inline int32_t distance_is_close_8u_C3R(uint8_t r1, uint8_t g1, uint8_t b1, uint8_t r2, uint8_t g2, uint8_t b2, uint32_t threshold)
{
    return (abs_uint(r1 - r2) + abs_uint(g1 - g2) + abs_uint(b1 - b2) <= 4.5 * threshold);
}

struct vibeModel_Sequential
{
    /* Parameters. */
    uint32_t width;
    uint32_t height;
    uint32_t numberOfSamples;
    uint32_t matchingThreshold;
    uint32_t matchingNumber;
    uint32_t updateFactor;
    
    /* Storage for the history. */
    uint8_t *historyImage;
    uint8_t *historyBuffer;
    uint32_t lastHistoryImageSwapped;
    
    /* Buffers with random values. */
    uint32_t *jump;
    int *neighbor;
    uint32_t *position;
};

// -----------------------------------------------------------------------------
// Print parameters
// -----------------------------------------------------------------------------
uint32_t libvibeModel_Sequential_PrintParameters(const vibeModel_Sequential_t *model)
{
    printf(
           "Using ViBe background subtraction algorithm\n"
           "  - Number of samples per pixel:       %03d\n"
           "  - Number of matches needed:          %03d\n"
           "  - Matching threshold:                %03d\n"
           "  - Model update subsampling factor:   %03d\n",
           libvibeModel_Sequential_GetNumberOfSamples(model),
           libvibeModel_Sequential_GetMatchingNumber(model),
           libvibeModel_Sequential_GetMatchingThreshold(model),
           libvibeModel_Sequential_GetUpdateFactor(model)
           );
    
    return(0);
}

// -----------------------------------------------------------------------------
// Creates the data structure
// -----------------------------------------------------------------------------
vibeModel_Sequential_t *libvibeModel_Sequential_New()
{
    /* Model structure alloc. */
    vibeModel_Sequential_t *model = NULL;
    model = (vibeModel_Sequential_t*)calloc(1, sizeof(*model));
    assert(model != NULL);
    
    /* Default parameters values. */
    model->numberOfSamples         = 20;
    model->matchingThreshold       = 3 ;
    model->matchingNumber          = 2;
    model->updateFactor            = 4;
    
    
    /* Storage for the history. */
    model->historyImage            = NULL;
    model->historyBuffer           = NULL;
    model->lastHistoryImageSwapped = 0;
    
    /* Buffers with random values. */
    model->jump                    = NULL;
    model->neighbor                = NULL;
    model->position                = NULL;
    
    return(model);
}

// -----------------------------------------------------------------------------
// Some "Get-ers"
// -----------------------------------------------------------------------------
uint32_t libvibeModel_Sequential_GetNumberOfSamples(const vibeModel_Sequential_t *model)
{
    assert(model != NULL); return(model->numberOfSamples);
}

uint32_t libvibeModel_Sequential_GetMatchingNumber(const vibeModel_Sequential_t *model)
{
    assert(model != NULL); return(model->matchingNumber);
}

uint32_t libvibeModel_Sequential_GetMatchingThreshold(const vibeModel_Sequential_t *model)
{
    assert(model != NULL); return(model->matchingThreshold);
}

uint32_t libvibeModel_Sequential_GetUpdateFactor(const vibeModel_Sequential_t *model)
{
    assert(model != NULL); return(model->updateFactor);
}

// -----------------------------------------------------------------------------
// Some "Set-ers"
// -----------------------------------------------------------------------------
int32_t libvibeModel_Sequential_SetMatchingThreshold(  vibeModel_Sequential_t *model, const uint32_t matchingThreshold)
{
    assert(model != NULL);
    assert(matchingThreshold > 0);
    model->matchingThreshold = matchingThreshold;
    return(0);
}

// -----------------------------------------------------------------------------
int32_t libvibeModel_Sequential_SetMatchingNumber( vibeModel_Sequential_t *model,const uint32_t matchingNumber)
{
    assert(model != NULL);
    assert(matchingNumber > 0);
    model->matchingNumber = matchingNumber;
    return(0);
}

// -----------------------------------------------------------------------------
int32_t libvibeModel_Sequential_SetUpdateFactor( vibeModel_Sequential_t *model, const uint32_t updateFactor)
{
    assert(model != NULL);
    assert(updateFactor > 0);
    model->updateFactor = updateFactor;
    /* We also need to change the values of the jump buffer ! */
    assert(model->jump != NULL);
    /* Shifts. */
    int size = (model->width > model->height) ? 2 * model->width + 1 : 2 * model->height + 1;
    for (int i = 0; i < size; ++i)
        model->jump[i] = (updateFactor == 1) ? 1 : (rand() % (2 * model->updateFactor)) + 1; // 1 or values between 1 and 2 * updateFactor.
    return(0);
}

// ----------------------------------------------------------------------------
// Frees the structure
// ----------------------------------------------------------------------------
int32_t libvibeModel_Sequential_Free(vibeModel_Sequential_t *model)
{
    if (model == NULL)
        return(-1);
    
    if (model->historyBuffer == NULL)
    {
        free(model);
        return(0);
    }
    
    free(model->historyImage);
    free(model->historyBuffer);
    free(model->jump);
    free(model->neighbor);
    free(model->position);
    free(model);
    
    return(0);
}

// -----------------------------------------------------------------------------
// Allocates and initializes a C1R model structure
// -----------------------------------------------------------------------------
int32_t libvibeModel_Sequential_AllocInit_8u_C1R(vibeModel_Sequential_t *model,const uint8_t *image_data,const uint32_t width,const uint32_t height)
{
    /* Some basic checks. */
    assert((image_data != NULL) && (model != NULL));
    assert((width > 0) && (height > 0));
    
    /* Finish model alloc - parameters values cannot be changed anymore. */
    model->width = width;
    model->height = height;
    
    /* Creates the historyImage structure. */
    model->historyImage = NULL;
    model->historyImage = (uint8_t*)malloc(NUMBER_OF_HISTORY_IMAGES * width * height * sizeof(*(model->historyImage)));
    
    assert(model->historyImage != NULL);
    
    
    /* Now creates and fills the history buffer. */
    model->historyBuffer = (uint8_t*)malloc(width * height * (model->numberOfSamples - NUMBER_OF_HISTORY_IMAGES) * sizeof(uint8_t));
    assert(model->historyBuffer != NULL);
    
    
    for (int i = 0; i < NUMBER_OF_HISTORY_IMAGES; ++i)
    {
        for (int index = width * height - 1; index >= 0; --index)
            model->historyImage[i * width * height + index] = image_data[index];
    }
    
    for (int index = width * height - 1; index >= 0; --index)
    {
        uint8_t value = image_data[index];
        for (int x = 0; x < (int)(model->numberOfSamples - NUMBER_OF_HISTORY_IMAGES); ++x)
        {
            
            int value_plus_noise = value + rand() % 2-1;
            if (value_plus_noise < 0)
            { value_plus_noise = 0; }
            if (value_plus_noise > 255)
            { value_plus_noise = 255; }
            model->historyBuffer[index * (model->numberOfSamples - NUMBER_OF_HISTORY_IMAGES) + x] = value_plus_noise;
        }
    }
    
    /* Fills the buffers with random values. */
    int size = (width > height) ? 2 * width + 1 : 2 * height + 1;
    model->jump = (uint32_t*)malloc(size * sizeof(*(model->jump)));
    assert(model->jump != NULL);
    
    model->neighbor = (int*)malloc(size * sizeof(*(model->neighbor)));
    assert(model->neighbor != NULL);
    
    model->position = (uint32_t*)malloc(size * sizeof(*(model->position)));
    assert(model->position != NULL);
    
    for (int i = 0; i < size; ++i)
    {
        model->jump[i] = (rand()% (2 * model->updateFactor)) + 1;            // Values between 1 and 2 * updateFactor.
        model->neighbor[i] = ((rand() % 3) - 1) + ((rand() % 3) - 1) * width; // Values between { -width - 1, ... , width + 1 }.
        model->position[i] = rand() % (model->numberOfSamples);               // Values between 0 and numberOfSamples - 1.
    }
    
    return(0);
}

// -----------------------------------------------------------------------------
// Segmentation of a C1R model
// -----------------------------------------------------------------------------
int32_t libvibeModel_Sequential_Segmentation_8u_C1R( vibeModel_Sequential_t *model, const uint8_t *image_data, uint8_t *segmentation_map,uint8_t *grayMap)
{
    
    /* Basic checks. */
    assert((image_data != NULL) && (model != NULL) && (segmentation_map != NULL));
    assert((model->width > 0) && (model->height > 0));
    assert(model->historyBuffer != NULL);
    assert((model->jump != NULL) && (model->neighbor != NULL) && (model->position != NULL));
    
    /* Some variables. */
    uint32_t width = model->width;
    uint32_t height = model->height;
    uint32_t matchingNumber = model->matchingNumber;
    uint32_t matchingThreshold = model->matchingThreshold;
    
    uint8_t *historyImage = model->historyImage;
    uint8_t *historyBuffer = model->historyBuffer;
    
    /* Segmentation. */
    memset(segmentation_map, matchingNumber - 1, width * height);
    
    
    // memset(segmentation_map_count,matchingNumber - 2, width * height);
    
    
    /* First history Image structure. */
    for (int index = width * height - 1; index >= 0; --index)
    {
        if (abs_uint(image_data[index] - historyImage[index]) > matchingThreshold)
            segmentation_map[index] = matchingNumber;
    }
    
    /* Next historyImages. */
    for (int i = 1; i < NUMBER_OF_HISTORY_IMAGES; ++i)
    {
        uint8_t *pels = historyImage + i * width * height;
        
        for (int index = width * height - 1; index >= 0; --index)
        {
            if (abs_uint(image_data[index] - pels[index]) <= matchingThreshold)
                --segmentation_map[index];
        }
    }
    
    /* For swapping. */
    model->lastHistoryImageSwapped = (model->lastHistoryImageSwapped + 1) % NUMBER_OF_HISTORY_IMAGES;
    uint8_t *swappingImageBuffer = historyImage + (model->lastHistoryImageSwapped) * width * height;//Ç°Á½Ö¡Í¼ÏñÉÏµÄÆäÖÐÒ»Ö¡
    
    /* Now, we move in the buffer and leave the historyImages. */
    int numberOfTests = (model->numberOfSamples - NUMBER_OF_HISTORY_IMAGES);
    
    for (int index = width * height - 1; index >= 0; --index)
    {
        if (segmentation_map[index] > 0)
        {
            /* We need to check the full border and swap values with the first or second historyImage.
             We still need to find a match before we can stop our search.
             */
            uint32_t indexHistoryBuffer = index * numberOfTests;
            uint8_t currentValue = image_data[index];
            
            for (int i = numberOfTests; i > 0; --i, ++indexHistoryBuffer)
            {
                if (abs_uint(currentValue - historyBuffer[indexHistoryBuffer]) <= matchingThreshold)
                {
                    --segmentation_map[index];
                    
                    /* Swaping: Putting found value in history image buffer. */
                    uint8_t temp = swappingImageBuffer[index];
                    swappingImageBuffer[index] = historyBuffer[indexHistoryBuffer];
                    historyBuffer[indexHistoryBuffer] = temp;
                    
                    
                    /* Exit inner loop. */
                    if (segmentation_map[index] <= 0) break;
                }
                
                /*	else
                 {
                 segmentation_map_copy[index]=segmentation_map_copy[index]+1;
                 
                 if(segmentation_map_copy[index]>20)
                 segmentation_map_copy[index]=-1;
                 
                 
                 }  */
                
            } // for
        } // if
    } // for
    
    /* Produces the output. Note that this step is application-dependent. */
    
    for (uint8_t *mask = segmentation_map; mask < segmentation_map + (width * height); ++mask)
    {
        if (*mask > 0)
            *mask = COLOR_FOREGROUND;
        
    }
    
    
    for (int index = width * height - 1; index >= 0; --index)
    {
        uint8_t *mask_copy = grayMap;
        int position=rand()%17;
        mask_copy[index]=historyBuffer[18*index+ position];
    }
    
    //detTime()
    
    return(0);
}

// ----------------------------------------------------------------------------
// Update a C1R model with Song proc
// ----------------------------------------------------------------------------
int32_t libvibeModel_Sequential_Update_8u_C1R_EX( vibeModel_Sequential_t *model,const cv::Mat& image_data, cv::Mat& updating_mask,cv::Mat& grayMap_count,cv::Mat& grayMap,MASK_PROC_CONFIG config)
{
    MASK_PRO_PROC proc =
    {
        updating_mask,
        grayMap_count,
    };
    maskProc(proc,config,image_data,grayMap);
    //return 0;
    return  libvibeModel_Sequential_Update_8u_C1R( model,image_data.data, updating_mask.data,proc.sobelRes.data,grayMap_count.data) ;
}

// ----------------------------------------------------------------------------
// Update a C1R model
// ----------------------------------------------------------------------------
int32_t libvibeModel_Sequential_Update_8u_C1R( vibeModel_Sequential_t *model,const uint8_t *image_data, uint8_t *updating_mask,uint8_t *BianJieJianCe,uint8_t *grayMap_count)
{
    /* Basic checks . */
    assert((image_data != NULL) && (model != NULL) && (updating_mask != NULL));
    assert((model->width > 0) && (model->height > 0));
    assert(model->historyBuffer != NULL);
    assert((model->jump != NULL) && (model->neighbor != NULL) && (model->position != NULL));
    
    
    /* Some variables. */
    uint32_t width = model->width;
    uint32_t height = model->height;
    
    uint8_t *historyImage = model->historyImage;
    uint8_t *historyBuffer = model->historyBuffer;
    
    /* Some utility variable. */
    int numberOfTests = (model->numberOfSamples - NUMBER_OF_HISTORY_IMAGES);
    
    /* Updating. */
    uint32_t *jump = model->jump;
    int *neighbor = model->neighbor;
    uint32_t *position = model->position;
    
    /* All the frame, except the border. */
    uint32_t shift, indX, indY;
    int x, y;
    
    for (y = 1; y < height - 1; ++y)
    {
        shift = rand() % width;
        indX = jump[shift];    // index_jump should never be zero (> 1).
        
        while (indX < width - 1)
        {
            int index = indX + y * width;
            
            
            if (updating_mask[index] == COLOR_BACKGROUND)
            {
                /* In-place substitution. */
                uint8_t value = image_data[index];
                int index_neighbor = index + neighbor[shift];
                
                if(grayMap_count[index]==200)
                {
                    for(int pp=0;pp<2;pp++)
                    {
                        historyImage[index + pp * width * height] = value;
                        historyImage[index_neighbor + pp * width * height] = value;
                    }
                    
                    for(int pp=0;pp<17;pp++)
                    {
                        historyBuffer[index * numberOfTests + pp] = value;
                        historyBuffer[index_neighbor * numberOfTests + pp] = value;
                    }
                }
                
                if (position[shift] < NUMBER_OF_HISTORY_IMAGES)
                {
                    
                    if (BianJieJianCe[index] < 100 )
                    {
                        historyImage[index + position[shift] * width * height] = value;
                        historyImage[index_neighbor + position[shift] * width * height] = value;
                    }
                    
                }
                else
                {
                    
                    int pos = position[shift] - NUMBER_OF_HISTORY_IMAGES;
                    
                    
                    
                    if (BianJieJianCe[index] < 100 )
                    {
                        historyBuffer[index * numberOfTests + pos] = value;
                        historyBuffer[index_neighbor * numberOfTests + pos] = value;
                    }
                }
            }
            
            if(grayMap_count[index]==200)
            {
                grayMap_count[index]=0;
                indX++;
            }
            else
            {
                ++shift;
                indX += jump[shift];
            }
            
            
            
        }
    }
    
    /* First row. */
    y = 0;
    shift = rand() % width;
    indX = jump[shift]; // index_jump should never be zero (> 1).
    
    while (indX <= width - 1)
    {
        int index = indX + y * width;
        
        if (updating_mask[index] == COLOR_BACKGROUND)
        {
            if (position[shift] < NUMBER_OF_HISTORY_IMAGES)
                historyImage[index + position[shift] * width * height] = image_data[index];
            else {
                int pos = position[shift] - NUMBER_OF_HISTORY_IMAGES;
                historyBuffer[index * numberOfTests + pos] = image_data[index];
            }
        }
        
        ++shift;
        indX += jump[shift];
    }
    
    /* Last row. */
    y = height - 1;
    shift = rand() % width;
    indX = jump[shift]; // index_jump should never be zero (> 1).
    
    while (indX <= width - 1)
    {
        int index = indX + y * width;
        
        if (updating_mask[index] == COLOR_BACKGROUND)
        {
            if (position[shift] < NUMBER_OF_HISTORY_IMAGES)
                historyImage[index + position[shift] * width * height] = image_data[index];
            else {
                int pos = position[shift] - NUMBER_OF_HISTORY_IMAGES;
                historyBuffer[index * numberOfTests + pos] = image_data[index];
            }
        }
        
        ++shift;
        indX += jump[shift];
    }
    
    /* First column. */
    x = 0;
    shift = rand() % height;
    indY = jump[shift]; // index_jump should never be zero (> 1).
    
    while (indY <= height - 1) {
        int index = x + indY * width;
        
        if (updating_mask[index] == COLOR_BACKGROUND) {
            if (position[shift] < NUMBER_OF_HISTORY_IMAGES)
                historyImage[index + position[shift] * width * height] = image_data[index];
            else {
                int pos = position[shift] - NUMBER_OF_HISTORY_IMAGES;
                historyBuffer[index * numberOfTests + pos] = image_data[index];
            }
        }
        
        ++shift;
        indY += jump[shift];
    }
    
    /* Last column. */
    x = width - 1;
    shift = rand() % height;
    indY = jump[shift]; // index_jump should never be zero (> 1).
    
    while (indY <= height - 1) {
        int index = x + indY * width;
        
        if (updating_mask[index] == COLOR_BACKGROUND) {
            if (position[shift] < NUMBER_OF_HISTORY_IMAGES )
                historyImage[index + position[shift] * width * height] = image_data[index];
            else {
                int pos = position[shift] - NUMBER_OF_HISTORY_IMAGES;
                historyBuffer[index * numberOfTests + pos] = image_data[index];
            }
        }
        
        ++shift;
        indY += jump[shift];
    }
    
    /* The first pixel! */
    if (rand() % model->updateFactor == 0)
    {
        if (updating_mask[0] == 0)
        {
            int position = rand() % model->numberOfSamples;
            
            if (position < NUMBER_OF_HISTORY_IMAGES)
                historyImage[position * width * height] = image_data[0];
            else {
                int pos = position - NUMBER_OF_HISTORY_IMAGES;
                historyBuffer[pos] =  image_data[0];
            }
        }
    }
    return(0);
}

// ----------------------------------------------------------------------------
// -------------------------- The same for C3R models -------------------------
// ----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Allocates and initializes a C3R model structure
// -----------------------------------------------------------------------------
int32_t libvibeModel_Sequential_AllocInit_8u_C3R(
                                                 vibeModel_Sequential_t *model,
                                                 const uint8_t *image_data,
                                                 const uint32_t width,
                                                 const uint32_t height
                                                 ) {
    /* Some basic checks. */
    assert((image_data != NULL) && (model != NULL));
    assert((width > 0) && (height > 0));
    
    /* Finish model alloc - parameters values cannot be changed anymore. */
    model->width = width;
    model->height = height;
    
    /* Creates the historyImage structure. */
    model->historyImage = NULL;
    model->historyImage = (uint8_t*)malloc(NUMBER_OF_HISTORY_IMAGES * (3 * width) * height * sizeof(uint8_t));
    assert(model->historyImage != NULL);
    
    for (int i = 0; i < NUMBER_OF_HISTORY_IMAGES; ++i) {
        for (int index = (3 * width) * height - 1; index >= 0; --index)
            model->historyImage[i * (3 * width) * height + index] = image_data[index];
    }
    
    assert(model->historyImage != NULL);
    /* Creates the history buffer. */
    model->historyBuffer = (uint8_t *)malloc((3 * width) * height * (model->numberOfSamples - NUMBER_OF_HISTORY_IMAGES) * sizeof(uint8_t));
    assert(model->historyBuffer != NULL);
    
    /* Fills the history buffer */
    for (int index = 0; index <  width * height; index++) {
        uint8_t value_C1 = image_data[3*index];
        uint8_t value_C2 = image_data[3*index + 1];
        uint8_t value_C3 = image_data[3*index + 2];
        
        for (int x = 0; x < model->numberOfSamples - NUMBER_OF_HISTORY_IMAGES; ++x) {
            /* Adds noise on the value */
            int value_plus_noise_C1 = value_C1 + rand() % 20 - 10;
            int value_plus_noise_C2 = value_C2 + rand() % 20 - 10;
            int value_plus_noise_C3 = value_C3 + rand() % 20 - 10;
            
            /* Limits the value + noise to the [0,255] range */
            if (value_plus_noise_C1 < 0)   { value_plus_noise_C1 = 0; }
            if (value_plus_noise_C1 > 255) { value_plus_noise_C1 = 255; }
            if (value_plus_noise_C2 < 0)   { value_plus_noise_C2 = 0; }
            if (value_plus_noise_C2 > 255) { value_plus_noise_C2 = 255; }
            if (value_plus_noise_C3 < 0)   { value_plus_noise_C3 = 0; }
            if (value_plus_noise_C3 > 255) { value_plus_noise_C3 = 255; }
            
            model->historyBuffer[index * 3 * (model->numberOfSamples - NUMBER_OF_HISTORY_IMAGES) + 3*x]     = value_plus_noise_C1;
            model->historyBuffer[index * 3 * (model->numberOfSamples - NUMBER_OF_HISTORY_IMAGES) + 3*x + 1] = value_plus_noise_C2;
            model->historyBuffer[index * 3 * (model->numberOfSamples - NUMBER_OF_HISTORY_IMAGES) + 3*x + 2] = value_plus_noise_C3;
        }
    }
    
    /* Fills the buffers with random values. */
    int size = (width > height) ? 2 * width + 1 : 2 * height + 1;
    
    model->jump = (uint32_t*)malloc(size * sizeof(*(model->jump)));
    assert(model->jump != NULL);
    
    model->neighbor = (int*)malloc(size * sizeof(*(model->neighbor)));
    assert(model->neighbor != NULL);
    
    model->position = (uint32_t*)malloc(size * sizeof(*(model->position)));
    assert(model->position != NULL);
    
    for (int i = 0; i < size; ++i) {
        model->jump[i] = (rand() % (2 * model->updateFactor)) + 1;            // Values between 1 and 2 * updateFactor.
        model->neighbor[i] = ((rand() % 3) - 1) + ((rand() % 3) - 1) * width; // Values between { width - 1, ... , width + 1 }.
        model->position[i] = rand() % (model->numberOfSamples);               // Values between 0 and numberOfSamples - 1.
    }
    
    return(0);
}

// -----------------------------------------------------------------------------
// Segmentation of a C3R model
// -----------------------------------------------------------------------------
int32_t libvibeModel_Sequential_Segmentation_8u_C3R(
                                                    vibeModel_Sequential_t *model,
                                                    const uint8_t *image_data,
                                                    uint8_t *segmentation_map
                                                    ) {
    /* Basic checks. */
    assert((image_data != NULL) && (model != NULL) && (segmentation_map != NULL));
    assert((model->width > 0) && (model->height > 0));
    assert(model->historyBuffer != NULL);
    assert((model->jump != NULL) && (model->neighbor != NULL) && (model->position != NULL));
    
    /* Some variables. */
    uint32_t width = model->width;
    uint32_t height = model->height;
    uint32_t matchingNumber = model->matchingNumber;
    uint32_t matchingThreshold = model->matchingThreshold;
    
    uint8_t *historyImage = model->historyImage;
    uint8_t *historyBuffer = model->historyBuffer;
    
    /* Segmentation. */
    memset(segmentation_map, matchingNumber - 1, width * height);
    
    /* First history Image structure. */
    uint8_t *first = historyImage;
    
    for (int index = width * height - 1; index >= 0; --index) {
        if (
            !distance_is_close_8u_C3R(
                                      image_data[3 * index], image_data[3 * index + 1], image_data[3 * index + 2],
                                      first[3 * index], first[3 * index + 1], first[3 * index + 2], matchingThreshold
                                      )
            )
            segmentation_map[index] = matchingNumber;
    }
    
    /* Next historyImages. */
    for (int i = 1; i < NUMBER_OF_HISTORY_IMAGES; ++i) {
        uint8_t *pels = historyImage + i * (3 * width) * height;
        
        for (int index = width * height - 1; index >= 0; --index) {
            if (
                distance_is_close_8u_C3R(
                                         image_data[3 * index], image_data[3 * index + 1], image_data[3 * index + 2],
                                         pels[3 * index], pels[3 * index + 1], pels[3 * index + 2], matchingThreshold
                                         )
                )
                --segmentation_map[index];
        }
    }
    
    // For swapping
    model->lastHistoryImageSwapped = (model->lastHistoryImageSwapped + 1) % NUMBER_OF_HISTORY_IMAGES;
    uint8_t *swappingImageBuffer = historyImage + (model->lastHistoryImageSwapped) * (3 * width) * height;
    
    // Now, we move in the buffer and leave the historyImages
    int numberOfTests = (model->numberOfSamples - NUMBER_OF_HISTORY_IMAGES);
    
    for (int index = width * height - 1; index >= 0; --index) {
        if (segmentation_map[index] > 0) {
            /* We need to check the full border and swap values with the first or second historyImage.
             * We still need to find a match before we can stop our search.
             */
            uint32_t indexHistoryBuffer = (3 * index) * numberOfTests;
            
            for (int i = numberOfTests; i > 0; --i, indexHistoryBuffer += 3) {
                if (
                    distance_is_close_8u_C3R(
                                             image_data[(3 * index)], image_data[(3 * index) + 1], image_data[(3 * index) + 2],
                                             historyBuffer[indexHistoryBuffer], historyBuffer[indexHistoryBuffer + 1], historyBuffer[indexHistoryBuffer + 2],
                                             matchingThreshold
                                             )
                    )
                    --segmentation_map[index];
                
                /* Swaping: Putting found value in history image buffer. */
                uint8_t temp_r = swappingImageBuffer[(3 * index)];
                uint8_t temp_g = swappingImageBuffer[(3 * index) + 1];
                uint8_t temp_b = swappingImageBuffer[(3 * index) + 2];
                
                swappingImageBuffer[(3 * index)]     = historyBuffer[indexHistoryBuffer];
                swappingImageBuffer[(3 * index) + 1] = historyBuffer[indexHistoryBuffer + 1];
                swappingImageBuffer[(3 * index) + 2] = historyBuffer[indexHistoryBuffer + 2];
                
                historyBuffer[indexHistoryBuffer]     = temp_r;
                historyBuffer[indexHistoryBuffer + 1] = temp_g;
                historyBuffer[indexHistoryBuffer + 2] = temp_b;
                
                /* Exit inner loop. */
                if (segmentation_map[index] <= 0) break;
            } // for
        } // if
    } // for
    
    /* Produces the output. Note that this step is application-dependent. */
    for (uint8_t *mask = segmentation_map; mask < segmentation_map + (width * height); ++mask)
        if (*mask > 0) *mask = COLOR_FOREGROUND;
    
    return(0);
}


// ----------------------------------------------------------------------------
// Update a C3R model
// ----------------------------------------------------------------------------
int32_t libvibeModel_Sequential_Update_8u_C3R(
                                              vibeModel_Sequential_t *model,
                                              const uint8_t *image_data,
                                              uint8_t *updating_mask
                                              ) {
    /* Basic checks. */
    assert((image_data != NULL) && (model != NULL) && (updating_mask != NULL));
    assert((model->width > 0) && (model->height > 0));
    assert(model->historyBuffer != NULL);
    assert((model->jump != NULL) && (model->neighbor != NULL) && (model->position != NULL));
    
    /* Some variables. */
    uint32_t width = model->width;
    uint32_t height = model->height;
    
    uint8_t *historyImage = model->historyImage;
    uint8_t *historyBuffer = model->historyBuffer;
    
    /* Some utility variable. */
    int numberOfTests = (model->numberOfSamples - NUMBER_OF_HISTORY_IMAGES);
    
    /* Updating. */
    uint32_t *jump = model->jump;
    int *neighbor = model->neighbor;
    uint32_t *position = model->position;
    
    /* All the frame, except the border. */
    uint32_t shift, indX, indY;
    int x, y;
    
    for (y = 1; y < height - 1; ++y) {
        shift = rand() % width;
        indX = jump[shift]; // index_jump should never be zero (> 1).
        
        while (indX < width - 1) {
            int index = indX + y * width;
            
            if (updating_mask[index] == COLOR_BACKGROUND) {
                /* In-place substitution. */
                uint8_t r = image_data[3 * index];
                uint8_t g = image_data[3 * index + 1];
                uint8_t b = image_data[3 * index + 2];
                
                int index_neighbor = 3 * (index + neighbor[shift]);
                
                if (position[shift] < NUMBER_OF_HISTORY_IMAGES) {
                    historyImage[3 * index + position[shift] * (3 * width) * height    ] = r;
                    historyImage[3 * index + position[shift] * (3 * width) * height + 1] = g;
                    historyImage[3 * index + position[shift] * (3 * width) * height + 2] = b;
                    
                    historyImage[index_neighbor + position[shift] * (3 * width) * height    ] = r;
                    historyImage[index_neighbor + position[shift] * (3 * width) * height + 1] = g;
                    historyImage[index_neighbor + position[shift] * (3 * width) * height + 2] = b;
                }
                else {
                    int pos = position[shift] - NUMBER_OF_HISTORY_IMAGES;
                    
                    historyBuffer[(3 * index) * numberOfTests + 3 * pos    ] = r;
                    historyBuffer[(3 * index) * numberOfTests + 3 * pos + 1] = g;
                    historyBuffer[(3 * index) * numberOfTests + 3 * pos + 2] = b;
                    
                    historyBuffer[index_neighbor * numberOfTests + 3 * pos    ] = r;
                    historyBuffer[index_neighbor * numberOfTests + 3 * pos + 1] = g;
                    historyBuffer[index_neighbor * numberOfTests + 3 * pos + 2] = b;
                }
            }
            
            ++shift;
            indX += jump[shift];
        }
    }
    
    /* First row. */
    y = 0;
    shift = rand() % width;
    indX = jump[shift]; // index_jump should never be zero (> 1).
    
    while (indX <= width - 1) {
        int index = indX + y * width;
        
        uint8_t r = image_data[3 * index];
        uint8_t g = image_data[3 * index + 1];
        uint8_t b = image_data[3 * index + 2];
        
        if (updating_mask[index] == COLOR_BACKGROUND) {
            if (position[shift] < NUMBER_OF_HISTORY_IMAGES) {
                historyImage[3 * index + position[shift] * (3 * width) * height    ] = r;
                historyImage[3 * index + position[shift] * (3 * width) * height + 1] = g;
                historyImage[3 * index + position[shift] * (3 * width) * height + 2] = b;
            }
            else {
                int pos = position[shift] - NUMBER_OF_HISTORY_IMAGES;
                
                historyBuffer[(3 * index) * numberOfTests + 3 * pos    ] = r;
                historyBuffer[(3 * index) * numberOfTests + 3 * pos + 1] = g;
                historyBuffer[(3 * index) * numberOfTests + 3 * pos + 2] = b;
            }
        }
        
        ++shift;
        indX += jump[shift];
    }
    
    /* Last row. */
    y = height - 1;
    shift = rand() % width;
    indX = jump[shift]; // index_jump should never be zero (> 1).
    
    while (indX <= width - 1) {
        int index = indX + y * width;
        
        uint8_t r = image_data[3 * index];
        uint8_t g = image_data[3 * index + 1];
        uint8_t b = image_data[3 * index + 2];
        
        if (updating_mask[index] == COLOR_BACKGROUND) {
            if (position[shift] < NUMBER_OF_HISTORY_IMAGES) {
                historyImage[3 * index + position[shift] * (3 * width) * height    ] = r;
                historyImage[3 * index + position[shift] * (3 * width) * height + 1] = g;
                historyImage[3 * index + position[shift] * (3 * width) * height + 2] = b;
            }
            else {
                int pos = position[shift] - NUMBER_OF_HISTORY_IMAGES;
                
                historyBuffer[(3 * index) * numberOfTests + 3 * pos    ] = r;
                historyBuffer[(3 * index) * numberOfTests + 3 * pos + 1] = g;
                historyBuffer[(3 * index) * numberOfTests + 3 * pos + 2] = b;
            }
        }
        
        ++shift;
        indX += jump[shift];
    }
    
    /* First column. */
    x = 0;
    shift = rand() % height;
    indY = jump[shift]; // index_jump should never be zero (> 1).
    
    while (indY <= height - 1) {
        int index = x + indY * width;
        
        uint8_t r = image_data[3 * index];
        uint8_t g = image_data[3 * index + 1];
        uint8_t b = image_data[3 * index + 2];
        
        if (updating_mask[index] == COLOR_BACKGROUND) {
            if (position[shift] < NUMBER_OF_HISTORY_IMAGES) {
                historyImage[3 * index + position[shift] * (3 * width) * height    ] = r;
                historyImage[3 * index + position[shift] * (3 * width) * height + 1] = g;
                historyImage[3 * index + position[shift] * (3 * width) * height + 2] = b;
            }
            else {
                int pos = position[shift] - NUMBER_OF_HISTORY_IMAGES;
                historyBuffer[(3 * index) * numberOfTests + 3 * pos    ] = r;
                historyBuffer[(3 * index) * numberOfTests + 3 * pos + 1] = g;
                historyBuffer[(3 * index) * numberOfTests + 3 * pos + 2] = b;
            }
        }
        
        ++shift;
        indY += jump[shift];
    }
    
    /* Last column. */
    x = width - 1;
    shift = rand() % height;
    indY = jump[shift]; // index_jump should never be zero (> 1).
    
    while (indY <= height - 1) {
        int index = x + indY * width;
        
        uint8_t r = image_data[3 * index];
        uint8_t g = image_data[3 * index + 1];
        uint8_t b = image_data[3 * index + 2];
        
        if (updating_mask[index] == COLOR_BACKGROUND) {
            if (position[shift] < NUMBER_OF_HISTORY_IMAGES) {
                historyImage[3 * index + position[shift] * (3 * width) * height    ] = r;
                historyImage[3 * index + position[shift] * (3 * width) * height + 1] = g;
                historyImage[3 * index + position[shift] * (3 * width) * height + 2] = b;
            }
            else {
                int pos = position[shift] - NUMBER_OF_HISTORY_IMAGES;
                
                historyBuffer[(3 * index) * numberOfTests + 3 * pos    ] = r;
                historyBuffer[(3 * index) * numberOfTests + 3 * pos + 1] = g;
                historyBuffer[(3 * index) * numberOfTests + 3 * pos + 2] = b;
            }
        }
        
        ++shift;
        indY += jump[shift];
    }
    
    /* The first pixel! */
    if (rand() % model->updateFactor == 0) {
        if (updating_mask[0] == 0) {
            int position = rand() % model->numberOfSamples;
            
            uint8_t r = image_data[0];
            uint8_t g = image_data[1];
            uint8_t b = image_data[2];
            
            if (position < NUMBER_OF_HISTORY_IMAGES) {
                historyImage[position * (3 * width) * height    ] = r;
                historyImage[position * (3 * width) * height + 1] = g;
                historyImage[position * (3 * width) * height + 2] = b;
            }
            else {
                int pos = position - NUMBER_OF_HISTORY_IMAGES;
                
                historyBuffer[3 * pos    ] = r;
                historyBuffer[3 * pos + 1] = g;
                historyBuffer[3 * pos + 2] = b;
            }
        }
    }
    
    return(0);
}

/**
 maskProc main algrothim body;
 */
uint32_t maskProc( MASK_PRO_PROC& proc,const MASK_PROC_CONFIG& config ,const cv::Mat& frame ,cv::Mat& grayMap)
{
    //cout << "hello" <<endl;
    vector< vector<cv::Point> > unicom_area_contours ;//´æ´¢ÁªÍ¨ÇøÓòÊýÁ¿
    static double count_for[720][1280]={0}; //´æ´¢Ç°¾°Á¬Ðø³öÏÖµÄÖ¡Êý
    static vector<cv::Point>  store_position[512];//·ÖÅäÒ»¸ö500´óÐ¡µÄdoubleÐÍÊý×é£¬ÓÃÓÚ´æ´¢Á¬Í¨ÇøÓòÎ»ÖÃ
    static double count_one[1000]={0};     //¼ÆËãÒ»¸öÇøÓòÄÚÇ°¾°µãÖÐÖ»³öÏÖN´ÎµÄµãÊý
    static double count_big[1000]={0};     //¼ÆËãÒ»¸öÇøÓòÄÚÇ°¾°µãÖÐÁ¬Ðø³öÏÖN´ÎµÄµãÊý
    static int famber[1000]={0};//´æ´¢¶ÔÓ¦Î»ÖÃ¹íÓ°³öÏÖµÄÖ¡Êý
    static cv::Mat one = cv::Mat::ones(frame.rows,frame.cols,CV_8UC1);                  //ËùÓÐµãÎª1µÄmatÍ¼Ïñ
    static cv::Mat zero =cv::Mat::zeros(frame.rows,frame.cols,CV_8UC1);    //ËùÓÐµãÎª0µÄmatÍ¼Ïñ

    static cv::Mat sobel_copy;             //´æ´¢±ßÔµ¼ì²âµÄ½á¹û
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));          
    cv::morphologyEx(proc.segmentationMap,proc.segmentationMap, cv::MORPH_OPEN, element);  //½øÐÐÐÎÌ¬Ñ§²Ù×÷ £¬¿ªÔËËã

    for(int col =0;col<frame.cols;col++)
    {
        for(int row  =0;row < frame.rows;row++)
        {
            if(proc.segmentationMap.at<uchar>(row,col)==255)
            {
                count_for[row][col]++;
            }
            else
            {
                count_for[row][col]=0; 
            }
        }
    }
    //cout << "hello1" <<endl;
    cv::Mat unicom_area;                   //ÓÃÓÚ¼ÆËãÁ¬Í¨ÇøÓò
    proc.segmentationMap.copyTo(unicom_area);
    int needUpdatePreTag = 0;                    //ÊÇ·ñÐèÒª¸üÐÂÇ°¾°µÄÅÐ¶Ï±êÖ¾
    int number=0;  //±³¾°»Ò¶È´óÓÚÇ°¾°»Ò¶È¸öÊý
    int Number=0;  //Ç°¾°»Ò¶È´óÓÚ±³¾°»Ò¶È¸öÊý
    int NUMBER=0;  // Ä³Ò»Á¬Í¨ÇøÓòÖÐËùÓÐµãµÄ¸öÊý
    int modelGrayVal = 0;       //±³¾°»Ò¶ÈÖµ
    int currentGrayVal = 0;       //Ç°¾°»Ò¶ÈÖµ
    
    cv::findContours(unicom_area,unicom_area_contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE); //²éÕÒÁªÍ¨ÇøÓò
    int max_contours=unicom_area_contours.size();                                             //ÁªÍ¨ÇøÓòÃæ»ý
    //cout << "hello2" <<endl;
    for(size_t i = 0; i < (size_t)max_contours; i++)  
    {   
        store_position[i]= unicom_area_contours[i];                                     
        cv::Rect Rect  = cv::boundingRect(store_position[i]);                           //Íâ½Ó¾ØÐÎ
        
        int x = Rect.x;
        int y = Rect.y;
        int width = Rect.width;
        int height = Rect. height;
        
        for( int col = x; col < width + x;col++ )
        {
            for( int row = y;row < height + y;row++ )
            {
                NUMBER++;
                modelGrayVal = grayMap.at<uchar>(row,col);
                currentGrayVal = frame.at<uchar>(row,col);
                
                if(modelGrayVal == 0)
                {
                    modelGrayVal = config.grayAlpha;
                }
                
                if(currentGrayVal == 0)
                {
                    currentGrayVal = config.grayAlpha;
                }
                
                if( MIN_GRAY_CHECK_VAL < currentGrayVal - modelGrayVal)
                {
                    number++;
                }
                if( MIN_GRAY_CHECK_VAL < modelGrayVal - currentGrayVal)
                {
                    Number++; 
                }
                
                if(count_for[row][col]> config.keepingTime )
                {
                    count_big[i]++;//¼ÆËãÒ»¸öÇøÓòÄÚÇ°¾°µãÖÐÁ¬Ðø³öÏÖN´ÎµÄµãÊý
                }
                
                if(count_for[row][col]==1)
                {
                    count_one[i]++;//¼ÆËãÒ»¸öÇøÓòÄÚÇ°¾°µãÖÐÖ»³öÏÖ1´ÎµÄµãÊý
                }
            }
        }
        
        if(config.isNeedRebuildForeground)
        {
            if(count_big[i] > 20 * count_one[i] && count_one[i] < 400)
            {			   
                needUpdatePreTag = 1;
            }
            else
            {
                needUpdatePreTag = 0;
            }
        }
        else
        {
            needUpdatePreTag  = 2;
        }
        //cout << "hello3" <<endl;
#if _DEBUG
        
#else
        cout << "sum pixel count : " << NUMBER <<endl;
#endif
        count_big[i]=0;
        count_one[i]=0;
        
        if(( number > 3 * Number && number > 100 ) || 1 == needUpdatePreTag )
        {
            //famber[i]++;
            NUMBER=1;
            number=0;
            Number=0;
            
            //if(famber[i] == 50 || needUpdatePreTag == 1)
            //{		
            //	famber[i]=0;
#if _DEBUG
            cout<<"  i  "<< i <<"  famber[i]  "<<famber[i]<<endl;
#endif
            for(int B = x;B < width + x;B++)
            {
                for(int C = y;C < height + y;C++)
                {
                    proc.segmentationMap.at<uchar>(C,B) = zero.at<uchar>(C,B);//ÕâÒ»²½ÊÇÏÂÒ»²½µÄ¹Ø¼ü
                    proc.grayMapCount.at<uchar>(C,B)= one.at<uchar>(C,B)*200;
                }
            }         
            //} 
        }
        else
        {   
            NUMBER=1;
            number=0;
            Number=0;
            famber[i]=0;
            
        }
    }
    //cout << "hello4" <<endl;
    sobel_copy = edge_detection(proc.segmentationMap);
    
    proc.sobelRes = sobel_copy;
    return 0;
}

cv::Mat edge_detection(cv::Mat frame_copy)
{
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    cv::Mat grad_x, grad_y,grad;
    cv::Mat abs_grad_x, abs_grad_y;
    cv::Mat JanCe_copy;
    frame_copy.copyTo(JanCe_copy);
    Sobel( JanCe_copy, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT ); // Çó X·½ÏòÌÝ¶È
    convertScaleAbs( grad_x, abs_grad_x );
    Sobel( JanCe_copy, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT ); // ÇóY·½ÏòÌÝ¶È
    convertScaleAbs( grad_y, abs_grad_y );
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, JanCe_copy );// ºÏ²¢ÌÝ¶È(½üËÆ)
    dilate(JanCe_copy,JanCe_copy,cv::Mat(5,5,CV_8U),cv::Point(-1,-1),2);
    imshow("sobel", JanCe_copy);
    return JanCe_copy;
}