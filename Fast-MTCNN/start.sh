#!/usr/bin/env bash
g++ mtcnn_opencv.cpp -o mtcnn_opencv -std=c++11 `pkg-config opencv --cflags --libs`