#ifndef COMMANDS_H
#define COMMANDS_H
#include <QApplication>
#include "GridWorld/toolsgw.h"
#include "Agents/qlearning.h"
#include "Agents/A2C/actorcritic.h"
#include "Agents/modelbased.h"
#include "GridWorld/episodeplayergw.h"

class Commands
{
 public:
  Commands();
  
  /////* GridWorld related commands */////

  //Map Generation
  
  void generateMapGW();
  void generateMapPoolGW();
  void showMapGW(int argc, char* argv[]);

  //QLearning

  void trainQLAgent();

  
  //Actor Advantage Critic 
  /*  
  void trainA2COneMapGW();
  void trainA2CMapPoolGW();
  void testA2C();
  void showCriticOnMapGW(int argc,char* argv[]);
  void showActorOnMapGW(int argc,char* argv[]);
  */
  //Model Based Planning
  
  void generateDataSetGW();
  void learnTransitionFunctionGW();
  void learnRewardFunctionGW();
  void testTransitionFunctionGW();
  //Other
  
  void test();
};

#endif // COMMANDS_H
