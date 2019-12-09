#include "plannergw.h"

PlannerGWImpl::PlannerGWImpl():
  usedDevice(torch::Device(torch::kCPU))
{
  if (torch::cuda::is_available())
    {
      std::cout << "CUDA detected for PlannerGW: training and inference will be done using CUDA." << std::endl;
      usedDevice = torch::Device(torch::kCUDA);
    }
  else
    {
      std::cout << "CUDA not available for PlannerGW: training and inference will be done using CPU." << std::endl;
    }
  this->to(usedDevice);
}

PlannerGWImpl::PlannerGWImpl(int size, int nConv, int nfc):
  usedDevice(torch::Device(torch::kCPU)), size(size), nConv(nConv), nfc(nfc), nLayers(-1+log(size)/log(2))
{
  init();
}

void PlannerGWImpl::init()
{
  if (torch::cuda::is_available())
    {
      std::cout << "CUDA detected for PlannerGW: training and inference will be done using CUDA." << std::endl;
      usedDevice = torch::Device(torch::kCUDA);
    }
  else
    {
      std::cout << "CUDA not available for PlannerGW: training and inference will be done using CPU." << std::endl;
    }
  this->to(usedDevice);
  convLayers.push_back(register_module("ConvLayer 1",torch::nn::Conv2d(torch::nn::Conv2dOptions(3,nConv,3).stride(1).padding(1))));
  for (int i=1;i<nLayers;i++)
    {
      convLayers.push_back(register_module("ConvLayer "+to_string(i+1),torch::nn::Conv2d(torch::nn::Conv2dOptions(nConv,nConv,3).stride(1).padding(1))));
    }
  fc = register_module("FC Layer",torch::nn::Linear(nConv*4,nfc));
  out = register_module("Output Layer", torch::nn::Linear(nfc,4));
}

torch::Tensor PlannerGWImpl::forward(torch::Tensor stateBatch)
{
  //Conversion to image if input is a batch of state vector
  
  bool imState = stateBatch.size(1)==3;        
  torch::Tensor x = stateBatch.clone();
  if (!imState)
    {
      x = ToolsGW().toRGBTensor(stateBatch.to(torch::Device(torch::kCPU))).to(usedDevice);            
    }

  //Forward Pass

   for (int i=0;i<nLayers;i++)
     {
       x = torch::relu(convLayers[i]->forward(x));
     }
   x = x.view({-1,4*nConv});
   x = torch::relu(fc->forward(x));
   x = torch::softmax(out->forward(x),1);
   return x;
}

void PlannerGWImpl::saveParams(std::string filename)
{
  std::ofstream f(filename);
  {
    f<<"###PARAMETERS FOR LOADING A PLANNER###"<<std::endl;
    f<<std::to_string(size)<<std::endl;
    f<<std::to_string(nConv)<<std::endl;
    f<<std::to_string(nfc)<<std::endl;
  }

}

void PlannerGWImpl::loadParams(std::string filename)
{
  std::ifstream f(filename);
  if (!f)
    {
      std::cout<<"An error has occured while trying to load the Transition model." << std::endl;
    }
  else {
    std::string line;
    std::getline(f,line);
    std::getline(f,line); size=stoi(line);
    std::getline(f,line); nConv=stoi(line);
    std::getline(f,line); nfc=stoi(line);
    nLayers = -1+log(size)/log(2);
    init();
  }  
}

torch::Device PlannerGWImpl::getUsedDevice()
{
  return usedDevice;
}
