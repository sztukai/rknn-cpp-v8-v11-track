set -e

# TARGET_SOC="rk3588"
GCC_COMPILER=aarch64-linux-gnu

export LD_LIBRARY_PATH=${TOOL_CHAIN}/lib64:$LD_LIBRARY_PATH
export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

# build
BUILD_DIR=${ROOT_PWD}/build/build_linux_aarch64

if [ ! -d "${BUILD_DIR}" ]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake ../.. -DCMAKE_SYSTEM_NAME=Linux
make -j8
make install
cd -

# v8
cd install/rknn_yolo_demo_Linux/ && ./rknn_yolo_demo ./model/RK3588/yolov8n.rknn ./model/RK3588/720p.mp4
# v11
# cd install/rknn_yolo_demo_Linux/ && ./rknn_yolo_demo ./model/RK3588/yolo11n.rknn ./model/RK3588/720p.mp4



