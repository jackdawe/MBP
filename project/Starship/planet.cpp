#include "planet.h"

Planet::Planet() {
}

Vect2d Planet::getCentre()
{
    return this->centre;
}

float Planet::getRadius()
{
    return this->radius;
}

float Planet::getMass()
{
    return this->mass;
}

void Planet::setCentre(Vect2d centre)
{
    this->centre = centre;
}

void Planet::setRadius(float radius)
{
    this->radius = radius;
}

void Planet::setMass(float mass)
{
    this->mass = mass;
}
