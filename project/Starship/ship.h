#ifndef SHIP_H
#define SHIP_H
#include "vect2d.h"

class Ship
{
    public:
        Ship();
        Vect2d getA();
        Vect2d getV();
        Vect2d getP();
        Vect2d getThrust();
        int getSignalColor();
        void setA(Vect2d a);
        void setV(Vect2d v);
        void setP(Vect2d p);
        void setThrust(Vect2d ft);
        void setSignalColor(int signalColor);
    private:
        Vect2d a;
        Vect2d v;
        Vect2d p;
        Vect2d thrust;
        int signalColor;




};

#endif // SHIP_H
