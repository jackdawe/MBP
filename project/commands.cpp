#include "commands.h"

Commands::Commands(){}

Commands::Commands(vector<string> parameters): parameters(parameters) {}

bool Commands::checkArguments(int nbParameters, vector<int> types)
{
  if (parameters.size()!=nbParameters)
    {
      cout<< "Error: this command takes " + to_string(nbParameters) + "  parameters but " + to_string(parameters.size()) + " were given" <<endl;
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
	      cout<<"Parameter " + to_string(i+1) + " has an invalid type (int)." << endl;
	      return false;
	    }
	case 1:
	  try
	    {
	      stof(parameters[i]);
	    }
	  catch (const std::invalid_argument& ia)
	    {
	      cout<<"Parameter " + to_string(i+1) + " has an invalid type (float)." << endl;
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
  MapGW map(stoi(parameters[0]));
  map.generate(stoi(parameters[1]));
  map.save(parameters[2]);
  return true;
}

bool Commands::generateMapPool()
{
  return true;
}
