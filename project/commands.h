#ifndef COMMANDS_H
#define COMMANDS_H
#include <QApplication>
#include "GridWorld/gridworld.h"
#include "Agents/qlearning.h"
#include "Agents/A2C/actorcritic.h"
#include "GridWorld/episodeplayergw.h"
#include <stdexcept>

class Commands
{
 public:
  Commands();
  
  //GridWorld related commands
  void generateMapGW();
  void generateMapPoolGW();
  void showMapGW(int argc, char* argv[]);
  void trainA2COneMapGW(); 
  void showCriticOnMapGW(int argc,char* argv[]);
  void showActorOnMapGW(int argc,char* argv[]);
};

#endif // COMMANDS_H
