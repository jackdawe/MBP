#include "forward.h"

ForwardImpl::ForwardImpl(): usedDevice(torch::Device(torch::kCPU))
{
  if (torch::cuda::is_available())
    {
      std::cout << "CUDA device detected: training and inference will be done using CUDA." << std::endl;
      usedDevice = torch::Device(torch::kCUDA);
    }
    else
    {
      std::cout << "CUDA device not available: training and inference will be done using CPU." << std::endl;
    }
  this->to(usedDevice);
}

void ForwardImpl::forward(torch::Tensor stateBatch, torch::Tensor actionBatch, bool restore){}

void ForwardImpl::computeLoss(torch::Tensor stateLabels, torch::Tensor rewardLabels){}  
