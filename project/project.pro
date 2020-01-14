QT -= gui
QT += widgets

QMAKE_CXX = ccache g++
CONFIG += c++11 console
CONFIG += debug
CONFIG -= app_bundle
WARN +=
# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
QMAKE_LFLAGS += -fopenmp -D_GLIBCXX_USE_CXX11_ABI=1 
QMAKE_CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=1 -fopenmp -g -ggdb $(WARN)
INCLUDEPATH += "$$PWD/../../libtorch/include"
INCLUDEPATH += "$$PWD/../../libtorch/include/torch/csrc/api/include"
LIBS += `pkg-config --libs opencv`
LIBS += -lstdc++fs
LIBS += -L"$$PWD/../../libtorch/lib" -ltorch -lc10 -lc10_cuda
LIBS += -L"/usr/local/cuda-9.2/lib64" -lcudart 
LIBS += -L"usr/include/" -lgflags


SOURCES += main.cpp \
    action.cpp \
    discreteaction.cpp \
    continuousaction.cpp \
    state.cpp \
    actionspace.cpp \
    world.cpp \
    forward.cpp \
    GridWorld/mapgw.cpp \
    GridWorld/gridworld.cpp \    
    GridWorld/episodeplayergw.cpp \    
    GridWorld/toolsgw.cpp \
    GridWorld/Models/convnetgw.cpp \
    GridWorld/Models/forwardgw.cpp \
    GridWorld/Models/plannergw.cpp \
    Starship/vect2d.cpp \
    Starship/planet.cpp \
    Starship/waypoint.cpp \
    Starship/ship.cpp \
    Starship/mapss.cpp \
    Starship/spaceworld.cpp \
    Starship/episodeplayerss.cpp \
    Starship/toolsss.cpp \
    Starship/Models/forwardss.cpp \
    agent.cpp \
    Agents/qlearning.cpp \
    Agents/actorcritic.cpp \
    Agents/modelbased.cpp \
    commands.cpp
    
HEADERS += \
    action.h \
    discreteaction.h \
    continuousaction.h \
    state.h \
    actionspace.h \
    world.h \
    forward.h \
    GridWorld/mapgw.h \
    GridWorld/gridworld.h \    
    GridWorld/episodeplayergw.h \    
    GridWorld/toolsgw.h \
    GridWorld/Models/convnetgw.h \
    GridWorld/Models/forwardgw.h \
    GridWorld/Models/plannergw.h \
    Starship/vect2d.h \
    Starship/planet.h \
    Starship/waypoint.h \
    Starship/ship.h \
    Starship/mapss.h \
    Starship/spaceworld.h \
    Starship/episodeplayerss.h \
    Starship/toolsss.h \
    Starship/Models/forwardss.h \
    agent.h \
    Agents/qlearning.h \
    Agents/actorcritic.h \
    Agents/modelbased.h \
    commands.h
    



