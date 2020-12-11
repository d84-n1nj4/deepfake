FROM nvcr.io/nvidia/pytorch:20.10-py3

WORKDIR /workspace

RUN apt update \
    && wget https://download.pytorch.org/libtorch/cu110/libtorch-cxx11-abi-shared-with-deps-1.7.0%2Bcu110.zip \
    && cd /workspace \
    && unzip libtorch-cxx11-abi-shared-with-deps-1.7.0+cu110.zip
