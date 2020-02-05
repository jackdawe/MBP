#include "ship.h"

Ship::Ship()
{
  a = Vect2d(0,0);
  v = Vect2d(0,0);
  p = Vect2d(0,0);
  width = 0;
  height = 0;
  thrust = Vect2d(0,0);
  signalColor = 0;
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

float Ship::getWidth() const
{
    return width;
}

void Ship::setWidth(float value)
{
    width = value;
}

float Ship::getHeight() const
{
    return height;
}

void Ship::setHeight(float value)
{
    height = value;
}
