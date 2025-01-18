#include <stdio.h>
#include <mutex>
#include "rknn_api.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "rkYolo.hpp"
#include "coreNum.hpp"

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; ++i)
    {
        shape_str += ", " + std::to_string(attr->dims[i]);
    }

    printf("index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
           "type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
           attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

static int saveFloat(const char *file_name, float *output, int element_size)
{
    FILE *fp;
    fp = fopen(file_name, "w");
    for (int i = 0; i < element_size; i++)
    {
        fprintf(fp, "%.6f\n", output[i]);
    }
    fclose(fp);
    return 0;
}

rkYoloModel::rkYoloModel(const std::string &model_path)
{
    this->model_path = model_path;
    nms_threshold = NMS_THRESH;      // 默认的NMS阈值
    box_conf_threshold = BOX_THRESH; // 默认的置信度阈值

    memset(&app_ctx, 0, sizeof(rknn_app_context_t));
}

int rkYoloModel::init(rknn_context *ctx_in, bool isChild)
{
    int ret = 0;
    // 读入模型数据
    rknn_context ctx = 0;
    printf("Loading model...\n");
    int model_data_size = 0;
    model_data = load_model(model_path.c_str(), &model_data_size);
    // 将数据传给NPU
    if (isChild == true)
    {
        ret = rknn_dup_context(ctx_in, &ctx);
        printf("************copy model **************\n");
    }
    else
    {
        ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
        printf("************init model **************\n");
    }
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    // 释放内存中的模型数据
    free(model_data);

    // 设置模型绑定的核心/Set the core of the model that needs to be bound
    rknn_core_mask core_mask;
    switch (get_core_num())
    {
    case 0:
        core_mask = RKNN_NPU_CORE_0;
        break;
    case 1:
        core_mask = RKNN_NPU_CORE_1;
        break;
    case 2:
        core_mask = RKNN_NPU_CORE_2;
        break;
    }
    ret = rknn_set_core_mask(ctx, core_mask);
    if (ret < 0)
    {
        printf("rknn_init core error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_output; i++)
    {
        printf("%d", io_num.n_output);
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
        // printf("**********%d dump_tensor_attr success***********\n", i);  
    }
    
    app_ctx.rknn_ctx = ctx;
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type == RKNN_TENSOR_INT8)
    {
        app_ctx.is_quant = true;
    }
    else
    {
        app_ctx.is_quant = false;
    }
    // printf("**********app_ctx.rknn_ctx success***********");    

    app_ctx.io_num = io_num;
    app_ctx.input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx.input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    // printf("**********app_ctx.input_attrs success***********");    

    app_ctx.output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx.output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));
    // printf("**********app_ctx.output_attrs success***********");    
    
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        app_ctx.model_channel = input_attrs[0].dims[1];
        app_ctx.model_height = input_attrs[0].dims[2];
        app_ctx.model_width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        app_ctx.model_height = input_attrs[0].dims[1];
        app_ctx.model_width = input_attrs[0].dims[2];
        app_ctx.model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx.model_height, app_ctx.model_width, app_ctx.model_channel);

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = app_ctx.model_width * app_ctx.model_height * app_ctx.model_channel;
    // printf("***********dump_tensor_attr success***********\n");  

    return 0;
}

void rkYoloModel::infer(cv::Mat &orig_img)
{
    std::lock_guard<std::mutex> lock(mtx);
    letterbox_t letter_box;
    img_width = orig_img.cols;
    img_height = orig_img.rows;

    object_detect_result_list od_results;
    memset(&letter_box, 0, sizeof(letterbox_t));
    memset(&od_results, 0x00, sizeof(object_detect_result_list));

    cv::Size target_size(app_ctx.model_width, app_ctx.model_height);
    cv::Mat resized_img(target_size.height, target_size.width, CV_8UC3, (0, 0, 0));
    // 计算缩放比例/Calculate the scaling ratio
    float scale_w = (float)target_size.width / orig_img.cols;
    float scale_h = (float)target_size.height / orig_img.rows;
    
    letter_box.scale = scale_w < scale_h ? scale_w : scale_h;
    letter_box.x_pad = int((target_size.width - letter_box.scale * orig_img.cols)/2);
    letter_box.y_pad = int((target_size.height - letter_box.scale * orig_img.rows)/2);

    // 图像缩放/Image scaling
    if (img_width != target_size.width  || img_height != target_size.height)
    {
        // rga
        rga_buffer_t src;
        rga_buffer_t dst;
        memset(&src, 0, sizeof(src));
        memset(&dst, 0, sizeof(dst));
        ret = resize_rga(src, dst, orig_img, resized_img, target_size);
        if (ret != 0)
        {
            fprintf(stderr, "resize with rga error\n");
        }
        inputs[0].buf = resized_img.data;
    }
    else
    {
        inputs[0].buf = orig_img.data;
    }

    ret = rknn_inputs_set(app_ctx.rknn_ctx, app_ctx.io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
    }

    ret = rknn_run(app_ctx.rknn_ctx, nullptr);

    rknn_output outputs[app_ctx.io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < app_ctx.io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = 0;
    }

    ret = rknn_outputs_get(app_ctx.rknn_ctx, app_ctx.io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
    }

    post_process(&app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, &od_results);

    rknn_outputs_release(app_ctx.rknn_ctx, app_ctx.io_num.n_output, outputs);

    // 画框和概率
    char text[256];
    std::vector<detect_result> objects;
    for (int i = 0; i < od_results.count; i++)
    {
        detect_result dr;
        object_detect_result *det_result = &(od_results.results[i]);
        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);

        // 打印预测物体的信息/Prints information about the predicted object
        // printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
        //        det_result->box.right, det_result->box.bottom, det_result->prop);
        // std::cout << "det_result->name: " << det_result->name << std::endl;

        // 如果类别为人
        if (det_result->cls_id == 0)
        {
            dr.classId = 0;
            dr.confidence = det_result->prop;
            dr.box.y = det_result->box.top;
            dr.box.x = det_result->box.left;
            dr.box.width = det_result->box.right - det_result->box.left;
            dr.box.height = det_result->box.bottom - det_result->box.top;
            objects.push_back(dr);
        }
        else
        {
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;
            rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(256, 0, 0, 256), 3);
            putText(orig_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
        }
    }
    std::vector<STrack> output_stracks = bytetracker.update(objects);
    for (unsigned long i = 0; i < output_stracks.size(); i++)
    {
        std::vector<float> tlwh = output_stracks[i].tlwh;
        bool vertical = tlwh[2] / tlwh[3] > 1.6;
        if (tlwh[2] * tlwh[3] > 20 && !vertical)
        {
            cv::Scalar s = bytetracker.get_color(output_stracks[i].track_id);
            cv::putText(orig_img, cv::format("%d people", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 5),
                    0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::rectangle(orig_img, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
        }
    }
}

rknn_context* rkYoloModel::get_pctx()
{
    return &app_ctx.rknn_ctx;
}

int rkYoloModel::release_yolo_model()
{
    if (app_ctx.input_attrs != NULL)
    {
        free(app_ctx.input_attrs);
        app_ctx.input_attrs = NULL;
    }
    if (app_ctx.output_attrs != NULL)
    {
        free(app_ctx.output_attrs);
        app_ctx.output_attrs = NULL;
    }
    if (app_ctx.rknn_ctx != 0)
    {
        rknn_destroy(app_ctx.rknn_ctx);
        app_ctx.rknn_ctx = 0;
    }
    return 0;
}

rkYoloModel::~rkYoloModel()
{
    if (app_ctx.input_attrs != NULL)
    {
        free(app_ctx.input_attrs);
        app_ctx.input_attrs = NULL;
    }
    if (app_ctx.output_attrs != NULL)
    {
        free(app_ctx.output_attrs);
        app_ctx.output_attrs = NULL;
    }
    if (app_ctx.rknn_ctx != 0)
    {
        rknn_destroy(app_ctx.rknn_ctx);
        app_ctx.rknn_ctx = 0;
    }
}