// File: train_cuda.h
#pragma once
#include <vector>
#include "loader.h"

#ifdef __CUDACC__
void train_cuda(const std::vector<Image>& dataset);
#endif
