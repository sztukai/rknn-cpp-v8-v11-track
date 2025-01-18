// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include "rga.h"
#include "im2d.h"
#include "RgaUtils.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"

int resize_rga(rga_buffer_t &src, rga_buffer_t &dst, const cv::Mat &image, cv::Mat &resized_image, const cv::Size &target_size)
{
    int ret = 0;
    im_rect dst_rect;
    memset(&dst_rect, 0, sizeof(dst_rect));
    rga_buffer_handle_t src_handle, dst_handle;

    int src_buf_size, dst_buf_size;
    int src_width, src_height, src_format;
    int dst_width, dst_height, dst_format;

    src_width = image.cols;
    src_height = image.rows;
    src_format = RK_FORMAT_BGR_888;
    if (image.type() != CV_8UC3)
    {
        printf("source image type is %d!\n", image.type());
        return -1;
    }

    dst_width = 640;
    dst_height = 640;
    dst_format = RK_FORMAT_RGB_888;

    src_buf_size = src_width * src_height * get_bpp_from_format(src_format);
    dst_buf_size = dst_width * dst_height * get_bpp_from_format(dst_format);

    src_handle = importbuffer_virtualaddr((void *)image.data, src_buf_size);
    dst_handle = importbuffer_virtualaddr((void *)resized_image.data, dst_buf_size);
    if (src_handle == 0 || dst_handle == 0) {
        ret = 0;
        printf("importbuffer failed!\n");
        goto release_buffer;
    }

    src = wrapbuffer_handle(src_handle, src_width, src_height, src_format);
    dst = wrapbuffer_handle(dst_handle, dst_width, dst_height, dst_format);

    dst_rect.x = 0;
    dst_rect.y = 140;
    dst_rect.width = 640;
    dst_rect.height = 360;

    ret = imcheck(src, dst, {}, dst_rect);
    if (IM_STATUS_NOERROR != ret) {
        printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        return -1;
    }

    ret = improcess(src, dst, {}, {}, dst_rect, {}, IM_SYNC);
    if (ret == IM_STATUS_SUCCESS) {
        ret = 0;
    } else {
        ret = -1;
        printf("running failed, %s\n", imStrError((IM_STATUS)ret));
        goto release_buffer;
    }

release_buffer:
    if (src_handle)
        releasebuffer_handle(src_handle);
    if (dst_handle)
        releasebuffer_handle(dst_handle);

    return ret;
}