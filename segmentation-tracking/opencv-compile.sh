# brew install cmake
# brew install gstreamer
# pip3 install numpy

# git clone https://github.com/opencv/opencv.git
# cd opencv
# git checkout master
# cd ..

# git clone https://github.com/opencv/opencv_contrib.git
# cd opencv_contrib
# git checkout master
# cd ..


mkdir build_opencv
cd build_opencv

cmake -S /Users/AkhilG/Desktop/hevctest/gstreamerTest/opencv -B /Users/AkhilG/Desktop/hevctest/gstreamerTest/build_opencv
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    # -D BUILD_opencv_python2=OFF \
    # -D BUILD_opencv_python3=ON \
    -D OPENCV_ENABLE_NONFREE:BOOL=ON \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D PYTHON_LIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
    -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    -D WITH_GSTREAMER=ON \ 
    -D WITH_OPENGL=ON \ 
    -D BUILD_EXAMPLES=ON ../opencv #downloaded opencv source folder

# make VERBOSE=1
sudo make -j$(sysctl -n hw.physicalcpu)
sudo make install

python3 -c "import cv2; print(cv2.__version__)"
