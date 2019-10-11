#ifndef WAYPOINT_H
#define WAYPOINT_H

#include "vect2d.h"

class Waypoint
{
    public:
        Waypoint();
        Vect2d getCentre();
        float getRadius();
        int getColor();
        void setCentre(Vect2d u);
        void setRadius(float r);
        void setColor(int color);

    private:
        Vect2d centre;
        float radius;
        int color;
};

#endif // WAYPOINT_H
