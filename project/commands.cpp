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


