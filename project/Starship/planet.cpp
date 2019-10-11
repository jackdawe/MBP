#include "planet.h"

Planet::Planet() {
}

Vect2d Planet::getCentre()
{
    return this->centre;
}

double Planet::getRadius()
{
    return this->radius;
}

double Planet::getMass()
{
    return this->mass;
}

void Planet::setCentre(Vect2d centre)
{
    this->centre = centre;
}

void Planet::setRadius(double radius)
{
    this->radius = radius;
}

void Planet::setMass(double mass)
{
    this->mass = mass;
}
