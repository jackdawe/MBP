QT -= gui
QT += widgets

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += main.cpp \
    action.cpp \
    discreteaction.cpp \
    continuousaction.cpp \
    state.cpp \
    agent.cpp \
    agenttrainer.cpp \
    GridWorld/mapgw.cpp \
    actionspace.cpp \
    controller.cpp \
    GridWorld/controllergw.cpp \
    Agents/qlearning.cpp \
    GridWorld/episodeplayergw.cpp \
    Starship/planet.cpp \
    Starship/waypoint.cpp \
    Starship/ship.cpp \
    Starship/vect2d.cpp \
    Starship/controllerss.cpp \
    Starship/episodeplayerss.cpp \
    Starship/mapss.cpp \
    Agents/randomagent.cpp

HEADERS += \
    action.h \
    discreteaction.h \
    continuousaction.h \
    state.h \
    agent.h \
    agenttrainer.h \
    GridWorld/mapgw.h \
    actionspace.h \
    controller.h \
    GridWorld/controllergw.h \
    Agents/qlearning.h \
    GridWorld/episodeplayergw.h \
    Starship/planet.h \
    Starship/waypoint.h \
    Starship/ship.h \
    Starship/vect2d.h \
    Starship/controllerss.h \
    Starship/episodeplayerss.h \
    Starship/mapss.h \
    Agents/randomagent.h
