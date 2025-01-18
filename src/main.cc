#include <stdio.h>
#include <memory>
#include <sys/time.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rkYolo.hpp"


int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Usage: %s <rknn model> <jpg> \n", argv[0]);
        return -1;
    }

    // 参数二，模型所在路径/The path where the model is located
    const std::string model_name = argv[1];

    // 参数三, 视频/摄像头
    std::string vedio_name = argv[2];

    init_post_process();

    rkYoloModel model(model_name);
    
    if (model.init(model.get_pctx(), false) != 0)
    {
        printf("model init fail!\n");
        return -1;
    }

    cv::VideoCapture capture;
    capture.open(vedio_name);
    
    struct timeval time;
    gettimeofday(&time, nullptr);
    auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    cv::Mat firstFrame;
    if (!capture.read(firstFrame)) {
        std::cerr << "无法读取第一帧" << std::endl;
        return -1;
    }
    cv::Size frameSize = firstFrame.size();

    cv::VideoWriter videoWriter("output.avi", cv::VideoWriter::fourcc('M','J','P','G'), 17, frameSize);
    if (!videoWriter.isOpened()) {
        std::cerr << "无法创建视频文件" << std::endl;
        return -1;
    }

    int frames = 0;
    auto beforeTime = startTime;
    while (capture.isOpened())
    {
        cv::Mat img;
        if (capture.read(img) == false)
            break;

        model.infer(img);

        videoWriter.write(img);
        frames++;

        if (frames % 120 == 0)
        {
            gettimeofday(&time, nullptr);
            auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
            printf("120帧内平均帧率:\t %f fps/s\n", 120.0 / float(currentTime - beforeTime) * 1000.0);
            beforeTime = currentTime;
        }
    }


    // videoWriter.release();
    gettimeofday(&time, nullptr);
    auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000;
    printf("Average:\t %f fps/s\n", float(frames) / float(endTime - startTime) * 1000.0);
    
    deinit_post_process();
    return 0;
}