#ifndef PLANET_H
#define PLANET_H

#include "vect2d.h"

class Planet
{
    public:
        Planet();
        Vect2d getCentre();
        double getRadius();
        double getMass();
        void setCentre(Vect2d centre);
        void setRadius(double radius);
        void setMass(double mass);

    private:
        Vect2d centre;
        double radius;
        double mass;

};

#endif // PLANET_H
