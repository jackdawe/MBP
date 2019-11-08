#include "commands.h"

Commands::Commands(){}

void Commands::generateMapGW()
{
  MapGW map(FLAGS_size);
  map.generate(FLAGS_maxObst);
  map.save(FLAGS_f);
}

void Commands::generateMapPoolGW()
{
  MapGW map(FLAGS_size); 
  map.generateMapPool(FLAGS_maxObst,FLAGS_dir,FLAGS_nmaps);
}

void Commands::showMapGW(int argc, char* argv[])
{
  QApplication a(argc,argv);
  EpisodePlayerGW ep(FLAGS_f);
  ep.showMap();
  a.exec();
}

void Commands::trainA2COneMapGW()
{
  GridWorld gw(FLAGS_f);
  int size = gw.getSize();
  ConvNetGW net(size,FLAGS_conv1,FLAGS_conv2,FLAGS_fc1);
  ParametersA2C params(FLAGS_g, FLAGS_lr, FLAGS_beta, FLAGS_zeta, FLAGS_bs, FLAGS_n);
  ActorCritic<GridWorld,ConvNetGW> agent(gw,net,params,true);
  agent.train();
  torch::save(agent.getModel(),"../model.pt");
}

void Commands::showCriticOnMapGW(int argc, char* argv[])
{
  QApplication a(argc,argv);
  vector<vector<string>> toDisplay;
  string filename = FLAGS_f;
  MapGW map;
  map.load(filename);
  int size = map.getSize();
  ConvNetGW net(8,16,16,128);
  torch::load(net,"../model.pt");
  for (int i=0;i<size;i++)
    {
      vector<string> line;
      for (int j=0;j<size;j++)
	{
	  GridWorld gw(filename,i,j);
	  gw.generateVectorStates();
	  gw.toRGBTensor(gw.getCurrentState().getStateVector());
	  torch::Tensor output = net->criticOutput(gw.toRGBTensor(gw.getCurrentState().getStateVector()).to(net->getUsedDevice()));
	  string val = to_string(*output.to(torch::Device(torch::kCPU)).data<float>());
	  string val2;
	  val2+=val[0],val2+=val[1],val2+=val[2],val2+=val[3],val2+=val[4];
	  line.push_back(val2);
	}
      toDisplay.push_back(line);
    }
  EpisodePlayerGW ep(FLAGS_f);
  ep.displayOnGrid(toDisplay);
  a.exec();
}

void Commands::showActorOnMapGW(int argc, char* argv[])
{
  QApplication a(argc,argv);
  vector<vector<string>> toDisplay;
  string filename = FLAGS_f;
  MapGW map;
  map.load(filename);
  int size = map.getSize();
  ConvNetGW net(8,16,16,128);
  torch::load(net,"../model.pt");
  for (int i=0;i<size;i++)
    {
      vector<string> line;
      for (int j=0;j<size;j++)
	{
	  GridWorld gw(filename,i,j);
	  gw.generateVectorStates();
	  gw.toRGBTensor(gw.getCurrentState().getStateVector());
	  torch::Tensor output = net->actorOutput(gw.toRGBTensor(gw.getCurrentState().getStateVector()).to(net->getUsedDevice()));
	  float max =0;
	  int iDirection;
	  string sDirection;
	  for (int k=0;k<4;k++)
	    {
	      float val = *output[0][k].to(torch::Device(torch::kCPU)).data<float>();
	      if (val>max)
                {
		  max = val;
		  iDirection = k;
                }
            }
	  switch (iDirection)
            {
            case 0:
	      sDirection = "UP";
	      break;
            case 1:
	      sDirection = "RIGHT";
	      break;
            case 2:
	      sDirection = "DOWN";
	      break;
            case 3:
	      sDirection = "LEFT";
	      break;
            }
	  line.push_back(sDirection);
	}
      toDisplay.push_back(line);
    }
  EpisodePlayerGW ep(FLAGS_f);
  ep.displayOnGrid(toDisplay);
  a.exec();
}


