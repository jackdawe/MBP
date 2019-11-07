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
  Commands(vector<string> parameters);
  bool checkArguments(int nbParameters,vector<int> types);
  
  //GridWorld related commands
  bool generateMapGW();
  bool generateMapPool();

 private:
  vector<string> parameters;
};

#endif // COMMANDS_H
