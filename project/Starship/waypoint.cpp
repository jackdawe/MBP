#include "waypoint.h"

Waypoint::Waypoint()
{
}

Vect2d Waypoint::getCentre()
{
    return this->centre;
}

double Waypoint::getRadius()
{
    return this->radius;
}

int Waypoint::getColor()
{
    return this->color;
}

void Waypoint::setCentre(Vect2d u)
{
    this->centre = u;
}

void Waypoint::setRadius(float r)
{
    this->radius = r;
}

void Waypoint::setColor(int color)
{
    this->color = color;
}
