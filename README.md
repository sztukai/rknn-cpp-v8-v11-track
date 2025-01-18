# 简介
### 此仓库为 RK3588 目标跟踪样例。</br>
该demo基于rkrga加速letterbox前处理，yolov8n/yolo11n rknn模型推理以及后处理，bytetracker目标跟踪，没有用线程加速。
![效果展示](v8_v11_track.gif)

# 使用说明
### 演示
  * 系统需安装有**OpenCV**
  * 系统需安装有**Eigen3** (sudo apt install libeigen3-dev / sudo cp -r /usr/local/include/eigen3/Eigen /usr/local/include)
  * 编译完成后进入install运行命令./rknn_yolo_demo **模型所在路径** **视频所在路径/摄像头序号**

### 部署应用
* 参考include/rkYolov5s.hpp中的rkYolov5s类构建rknn模型类

### 测试模型来源: 
* v8: 基于coco128数据集进行110-120轮的训练，后基于rknn_toolkit2转换为rknn模型。
* v11: 基于coco128数据集进行160-170轮的训练，后基于rknn_toolkit2转换为rknn模型。

# 补充
* 异常处理尚未完善, 目前仅支持rk3588/rk3588s/rk3588j下的运行。

# Acknowledgements
* https://github.com/ifzhang/ByteTrack
* https://github.com/ultralytics/ultralytics
* https://github.com/rockchip-linux/rknpu2
* https://github.com/shaoshengsong/DeepSORT
* https://github.com/airockchip/rknn_model_zoo