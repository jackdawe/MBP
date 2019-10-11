#ifndef PLANET_H
#define PLANET_H

#include "vect2d.h"

class Planet
{
    public:
        Planet();
        Vect2d getCentre();
        float getRadius();
        float getMass();
        void setCentre(Vect2d centre);
        void setRadius(float radius);
        void setMass(float mass);

    private:
        Vect2d centre;
        float radius;
        float mass;

};

#endif // PLANET_H
