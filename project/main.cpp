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
  else if (FLAGS_cmd == "gwqltr")
    {
      c.trainQLAgentGW();
    }
  else if (FLAGS_cmd == "gwqlpl")
    {
      c.playQLAgentGW(argc,argv);
    }
  else if (FLAGS_cmd == "gwqleval")
    {
      c.evaluateQLPolicy(argc,argv);
    }    
  else if (FLAGS_cmd == "gwa2ctr")
    {
      c.trainA2CGW();
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
  else if(FLAGS_cmd == "gwmbfm")
    {
      c.learnForwardModelGW();
    }
  else if(FLAGS_cmd == "gwmbpn")
    {
      //      c.trainPolicyNetGW();
    }
  else if(FLAGS_cmd == "gwmbplay")
    {
      c.playModelBasedGW(argc,argv);
    }
  else if(FLAGS_cmd == "tc1")
    {
      c.tc1();
    }
  else if(FLAGS_cmd=="ssmgen")    
    {
      c.generateMapSS();
    }
  else if(FLAGS_cmd=="ssmpgen")
    {
      c.generateMapPoolSS();
    }
  else if(FLAGS_cmd=="ssmshow")
    {
      c.showMapSS(argc,argv);
    }
  else if(FLAGS_cmd=="sspr")
    {
      c.playRandomSS(argc,argv);
    }
  else if(FLAGS_cmd=="ssdsgen")
    {
      c.generateDataSetSS();
    }
  else if(FLAGS_cmd=="ssfmtr")
    {
      c.trainForwardModelSS();
    }
  else if(FLAGS_cmd=="ssfmte")
    {
      c.testForwardModelSS();
    }
  else if(FLAGS_cmd=="ssaof")
    {
      c.actionOverfitSS();
    }
  else if(FLAGS_cmd=="sssgen")
    {
      c.generateSeedSS();
    }
  else if(FLAGS_cmd=="ssmbplay")
    {
      c.playPlannerSS(argc,argv);
    }
  else if(FLAGS_cmd=="ssmbtest")
    {
      c.testPlannerSS();
    }
  else
    {
      cout<<"Invalid command. Please refer to github README file for valid commands"<<endl;
    }      
}
