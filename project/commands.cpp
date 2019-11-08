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

bool Commands::generateMapPool()
{
  if (!checkArguments(4,{0,0,2,0}))
    {
      return false;
    }
  MapGW map(stoi(parameters[0])); //Size of the map
  map.generateMapPool(stoi(parameters[1]),parameters[2],stoi(parameters[3])); //Max number of obstacles, path to the folder in which to create the mapPool, number of maps to be generated
  return true;
}
