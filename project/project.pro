QT -= gui
QT += widgets

QMAKE_CXX = ccache g++
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
QMAKE_LFLAGS += "-Wl,-rpath,\'\$$ORIGIN/../..\'"
QMAKE_CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0
INCLUDEPATH += /home/jack/Documents/pytorchTest/libtorch/include
INCLUDEPATH += /home/jack/Documents/pytorchTest/libtorch/include/torch/csrc/api/include
INCLUDEPATH += /home/opencv/include
LIBS += -L/home/opencv/build/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc
LIBS += -L/home/jack/Documents/pytorchTest/libtorch/lib
LIBS += -ltorch -lc10

SOURCES += main.cpp \
    action.cpp \
    discreteaction.cpp \
    continuousaction.cpp \
    state.cpp \
    agent.cpp \
    GridWorld/mapgw.cpp \
    actionspace.cpp \
    Agents/qlearning.cpp \
    GridWorld/episodeplayergw.cpp \
    Starship/planet.cpp \
    Starship/waypoint.cpp \
    Starship/ship.cpp \
    Starship/vect2d.cpp \
    Starship/episodeplayerss.cpp \
    Starship/mapss.cpp \
    Agents/A2C/actorcritic.cpp \
    Agents/A2C/parametersa2c.cpp \
    world.cpp \
    GridWorld/gridworld.cpp \
    Starship/spaceworld.cpp \
    GridWorld/Models/convnetgw.cpp \
    GridWorld/Models/modela2cgw.cpp

HEADERS += \
    action.h \
    discreteaction.h \
    continuousaction.h \
    state.h \
    agent.h \
    GridWorld/mapgw.h \
    actionspace.h \
    Agents/qlearning.h \
    GridWorld/episodeplayergw.h \
    Starship/planet.h \
    Starship/waypoint.h \
    Starship/ship.h \
    Starship/vect2d.h \
    Starship/episodeplayerss.h \
    Starship/mapss.h \
    Agents/A2C/actorcritic.h \
    Agents/A2C/parametersa2c.h \
    world.h \
    GridWorld/gridworld.h \
    Starship/spaceworld.h \
    GridWorld/Models/convnetgw.h \
    GridWorld/Models/modela2cgw.h




