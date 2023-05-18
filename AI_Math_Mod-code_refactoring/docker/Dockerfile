# 1. Use an official Python runtime as a parent image
FROM python:3.5

# Install ubuntu packages
RUN apt-get update && apt-get install -y --no-install-recommends \
git \
libboost-dev \
libboost-python-dev \
libatlas-dev \
libatlas-base-dev \
libgoogle-glog-dev \
libhdf5-dev \
protobuf-compiler \
libsnappy-dev \
libleveldb-dev \
liblmdb-dev \
libboost-system-dev \
libboost-thread-dev \
libboost-filesystem-dev \
autoconf \
vim \
libjsoncpp-dev \
cmake \
binutils-dev \
libtool && \
cd /usr/lib/x86_64-linux-gnu && \
ln -s libhdf5_serial.so libhdf5.so && \
ln -s libhdf5_serial_hl.so libhdf5_hl.so && \
ln -s libboost_python-py35.so libboost_python3.so

# 2. Set the working directory to /app
WORKDIR /workspace


CMD ["bash"]