#include "commands.h"

Commands::Commands(){}

Commands::Commands(vector<string> parameters): parameters(parameters) {}

bool Commands::checkArguments(int nbParameters, vector<int> types)
{
  if (parameters.size()!=nbParameters)
    {
      cout<< "Error: this command takes " + to_string(nbParameters) + " parameters but " + to_string(parameters.size()) + " were given" <<endl;
      return false;
    }
  for (int i=0;i<nbParameters;i++)
    {
      switch (types[i])
	{
	case 0:
	  try
	    {
	      stoi(parameters[i]);
	    }
	  catch (const std::invalid_argument& ia)
	    {
	      cout<<"Parameter " + to_string(i+1) + " has an invalid type (int expected)." << endl;
	      return false;
	    }
	case 1:
	  try
	    {
	      stof(parameters[i]);
	    }
	  catch (const std::invalid_argument& ia)
	    {
	      cout<<"Parameter " + to_string(i+1) + " has an invalid type (float expected)." << endl;
	      return false;
	    }
	}
    }
  return true;
}

bool Commands::generateMapGW()
{
  if (!checkArguments(3,{0,0,2}))
    {
      return false;
    }
  MapGW map(stoi(parameters[0])); //Expected to be the size of the map
  map.generate(stoi(parameters[1])); //Expected to be the maximum number of obstacles
  map.save(parameters[2]); //Expected to be the path to the file to be created
  return true;
}

bool Commands::generateMapPoolGW()
{
  if (!checkArguments(4,{0,0,2,0}))
    {
      return false;
    }
  MapGW map(stoi(parameters[0])); //Size of the map
  map.generateMapPool(stoi(parameters[1]),parameters[2],stoi(parameters[3])); //Max number of obstacles, path to the folder in which to create the mapPool, number of maps to be generated
  return true;
}

bool Commands::showMapGW(int argc, char* argv[])
{
  QApplication a(argc,argv);
  EpisodePlayerGW ep(parameters[0]);
  ep.showMap();
  a.exec();
  return true;
}

bool Commands::trainA2COneMapGW()
{
  if (!checkArguments(7,{2,1,1,1,1,0,0}))
    {
      return false;
    }
  GridWorld gw(parameters[0]); //Path to the map on which to train 
  int size = gw.getSize();
  ConvNetGW net(size,16,16,size*size*2);
  TORCH_MODULE(net);
  //Discount factor, Learning Rate, EntropyLoss coefficient, ValueLoss coefficient, Batch size, Number of episodes 
  ParametersA2C params(stof(parameters[1]),stof(parameters[2]), stof(parameters[3]),stof(parameters[4]), stoi(parameters[5]), stoi(parameters[6]));
  ActorCritic<GridWorld,ConvNetGW> agent(gw,net,params,true);
  agent.train();
  torch::save(agent.getModel(),"../model.pt");
}


