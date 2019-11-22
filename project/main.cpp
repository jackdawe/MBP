#include "commands.h"
DEFINE_string(cmd,"help","Command to execute");

int main(int argc, char *argv[])
{
  gflags::ParseCommandLineFlags(&argc,&argv,true);
  Commands c;
  if (FLAGS_cmd == "help")
    {
      cout<<"No help yet... Unlucky !"<<endl;
    }
  else if (FLAGS_cmd == "gwmgen")
    {
      c.generateMapGW();
    }
  else if (FLAGS_cmd == "gwmpgen")
    {
      c.generateMapPoolGW();
    }
  else if (FLAGS_cmd == "gwmshow")
    {
      c.showMapGW(argc,argv);
    }
  else if (FLAGS_cmd == "gwa2c1m")
    {
      c.trainA2COneMapGW();
    }
  else if (FLAGS_cmd == "gwa2cmp")
    {
      c.trainA2CMapPoolGW();
    }
  else if(FLAGS_cmd == "gwa2cvshow")
    {
      c.showCriticOnMapGW(argc,argv);
    }
  else if(FLAGS_cmd == "gwa2cpshow")
    {
      c.showActorOnMapGW(argc,argv);
    }
  else if(FLAGS_cmd == "gwa2ctest")
    {
      c.testA2C();
    }
  else if(FLAGS_cmd == "gwdsgen")
    {
      c.generateDataSetGW();
    }
  else if(FLAGS_cmd == "gwmbft")
    {
      c.learnTransitionFunctionGW();
    }
  else if(FLAGS_cmd == "gwmbfr")
    {
      c.learnRewardFunctionGW();
    }
  else if(FLAGS_cmd == "gwmbftte")
    {
      c.testTransitionFunctionGW();
    }  
  else if(FLAGS_cmd == "test")
    {
      c.test();
    }
  else
    {
      cout<<"Invalid command. Please refer to github README file for valid commands"<<endl;
    }
}
