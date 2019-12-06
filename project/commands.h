#ifndef COMMANDS_H
#define COMMANDS_H
#include <QApplication>
#include "GridWorld/toolsgw.h"
#include "Agents/qlearning.h"
#include "Agents/A2C/actorcritic.h"
#include "Agents/modelbased.h"
#include "Agents/modelbased2.h"
#include "GridWorld/episodeplayergw.h"
#include "Starship/episodeplayerss.h"

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

  void trainQLAgentGW();
  void playQLAgentGW(int argc, char* argv[]);
  void evaluateQLPolicy(int argc, char* argv[]);
  
  //Actor Advantage Critic 
    
  void trainA2CGW();  
  void testA2C();
  void showCriticOnMapGW(int argc,char* argv[]);
  void showActorOnMapGW(int argc,char* argv[]);
  
  //Model Based Planning
  
  void generateDataSetGW();
  void learnTransitionFunctionGW();
  void learnRewardFunctionGW();
  void learnForwardModelGW();

  /////* Starship related commands */////

  void generateMapSS();
  void showMapSS(int argc, char* argv[]);
  
  //Other
  
  void test();
  void test2();
};

#endif // COMMANDS_H
