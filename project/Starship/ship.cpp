#include "ship.h"

Ship::Ship()
{

}

Vect2d Ship::getA()
{
    return this->a;
}

Vect2d Ship::getV()
{
    return this->v;
}

Vect2d Ship::getP()
{
    return this->p;
}

Vect2d Ship::getThrust()
{
    return this->thrust;
}

int Ship::getSignalColor()
{
    return signalColor;
}

void Ship::setA(Vect2d a)
{
    this->a = a;
}

void Ship::setV(Vect2d v)
{
    this->v = v;
}

void Ship::setP(Vect2d p)
{
    this->p = p;
}

void Ship::setThrust(Vect2d ft)
{
    this->thrust = ft;
}

void Ship::setSignalColor(int signalColor)
{
    this->signalColor=signalColor;
}

int Ship::getWidth() const
{
    return width;
}

void Ship::setWidth(int value)
{
    width = value;
}

int Ship::getHeight() const
{
    return height;
}

void Ship::setHeight(int value)
{
    height = value;
}
