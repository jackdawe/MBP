#include "vect2d.h"

Vect2d::Vect2d(): x(0),y(0)
{

}

Vect2d::Vect2d(float x,float y)
{
    this->x = x;
    this->y = y;
}

Vect2d Vect2d::sum(Vect2d v) //sum(u,v) returns the vector u+v
{
    Vect2d w(x+v.x,y+v.y);
    return w;
}

Vect2d Vect2d::dilate(float k) //dilate(u,k) returns the vector k*u
{
    return Vect2d(k*x,k*y);
}

float Vect2d::scalarProduct(Vect2d v) //returns the scalar product between two vectors
{
    return (x*v.x+y*v.y);
}

float Vect2d::norm() //returns the norm of the vector
{
    return (sqrt(scalarProduct(*this)));
}

float Vect2d::angle() //returns the angle in radians ranging from 0 to 2*pi between the vector and an arbitrary origin vector
{
  if ( (y/norm()) >=0 )
    {
      return acos(x/norm());
    }
  else
    {
      return -acos(x/norm());
    }
}

float Vect2d::distance(Vect2d v) //computes the distance between two vectors
{
    return (this->sum(v.dilate(-1)).norm());
}

