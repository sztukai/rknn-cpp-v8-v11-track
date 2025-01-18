#ifndef _RKYOLO_HPP_
#define _RKYOLO_HPP_

#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "STrack.h"
#include "preprocess.h"
#include "postprocess.h"
#include "rknn_api.h"
#include "BYTETracker.h" 

static void dump_tensor_attr(rknn_tensor_attr *attr);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);

class rkYoloModel
{
private:
    int ret;
    std::mutex mtx;
    std::string model_path;
    unsigned char *model_data;      //
    rknn_app_context_t app_ctx;    //

    rknn_input inputs[1];
    BYTETracker bytetracker;
    int img_width, img_height;
    rknn_input_output_num io_num;
    float nms_threshold, box_conf_threshold;

public:
    ~rkYoloModel();
    int release_yolo_model();
    rknn_context *get_pctx();
    void infer(cv::Mat &ori_img);
    rkYoloModel(const std::string &model_path);
    int init(rknn_context *ctx_in, bool isChild);
};

#endif