#ifndef _RKNN_YOLO_DEMO_POSTPROCESS_H_
#define _RKNN_YOLO_DEMO_POSTPROCESS_H_
#include <stdint.h>
#include <vector>
#include "rknn_api.h"

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.2
#define BOX_THRESH 0.4

typedef struct {
    int x_pad;
    int y_pad;
    float scale;
} letterbox_t;

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
#if defined(ZERO_COPY)  
    rknn_tensor_mem* input_mems[1];
    rknn_tensor_mem* output_mems[9];
    rknn_tensor_attr* input_native_attrs;
    rknn_tensor_attr* output_native_attrs;
#endif
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_app_context_t;

typedef struct {
    int left;
    int top;
    int right;
    int bottom;
} image_rect_t;

typedef struct 
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct 
{
    image_rect_t box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct 
{
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

int init_post_process();
void deinit_post_process();
char *coco_cls_to_name(int cls_id);
int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results);
#endif //_RKNN_YOLOV5_DEMO_POSTPROCESS_H_
