#include "commands.h"

int main(int argc, char *argv[])
{
  string command = argv[1];
  vector<string> parameters(argv+2,argv+argc);
  Commands c(parameters);
  if (command == "gwmgen")
    {
      c.generateMapGW();
    }
  else if (command == "gwmpgen")
    {
      c.generateMapPoolGW();
    }
  else if (command == "gwmshow")
    {
      c.showMapGW(argc,argv);
    }
  else if (command == "a2c1map")
    {
      c.trainA2COneMapGW();
    }
  else
    {
      cout<<"Invalid command. Please refer to github README file for valid commands"<<endl;
    }
  
  
  /*
  
  //MapGW mapgen(8);
  //mapgen.generateMapPool(5,"../GridWorld_Maps/TestEasy8x8/",5);
  //cout<<"hey"<<endl;
  float gamma = stof(argv[1]);
  float learningRate = stof(argv[2]);
  float beta = stof(argv[3]);
  float zeta = stof(argv[4]);
  int batchSize = stoi(argv[5]);
  int nEpisodes = stoi(argv[6]);
  //LOADING MAP AND TRAINING AGENT
  string mapTag = "../GridWorld_Maps/TestEasy8x8/";
  GridWorld gw(mapTag,5);
  int size = gw.getSize();
  ConvNetGW net(size,16,16,size*size*2);
  ParametersA2C params(gamma,learningRate,beta,zeta,batchSize,nEpisodes);
  ActorCritic<GridWorld,ConvNetGW> agent(gw,net,params,true);
  agent.train();
  //SHOWING THE POLICY
  QApplication a(argc, argv);
  vector<vector<string>> texts;
  vector<vector<string>> texts2;
  mapTag = mapTag + "map0";
  for (int i=0;i<size;i++)
    {
      vector<string> textsL(size,"");
      vector<string> textsL2(size,"");
      for (int j=0;j<size;j++)
        {
	  GridWorld gw2(mapTag,i,j);	
	  gw2.generateVectorStates();
	  torch::Tensor output = agent.getModel().actorOutput(gw2.toRGBTensor(gw2.getCurrentState().getStateVector()).to(agent.getModel().getUsedDevice()));
	  torch::Tensor output2 = agent.getModel().criticOutput(gw2.toRGBTensor(gw2.getCurrentState().getStateVector()).to(agent.getModel().getUsedDevice()));
	  float max = 0;
	  int dir;
	  string sdir;
	  for (int k=0;k<4;k++)
            {
	      if (*output[0][k].to(torch::Device(torch::kCPU)).data<float>()>max)
                {
		  max = *output[0][k].to(torch::Device(torch::kCPU)).data<float>();
		  dir = k;
                }
            }
	  switch (dir)
            {
            case 0:
	      sdir = "UP";
	      break;
            case 1:
	      sdir = "RIGHT";
	      break;
            case 2:
	      sdir = "DOWN";
	      break;
            case 3:
	      sdir = "LEFT";
	      break;
            }
	  textsL[j] = sdir;
	  
	  string val = to_string(*output2.to(torch::Device(torch::kCPU)).data<float>());
	  string val2;
	  val2+=val[0],val2+=val[1],val2+=val[2],val2+=val[3],val2+=val[4];
	  textsL2[j] = val2;
        }
        texts.push_back(textsL);
        texts2.push_back(textsL2);
    }
  cout<<"hey"<<endl;
  EpisodePlayerGW ep(mapTag);
  ep.displayOnGrid(texts);
  
  a.exec();
  */
}
