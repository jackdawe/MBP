#ifndef VECT2D_H
#define VECT2D_H
#include <math.h>

class Vect2d
{
    public:
        Vect2d();
        Vect2d(float x,float y);
        Vect2d sum(Vect2d v);
        Vect2d dilate(float k);
        float scalarProduct(Vect2d v);
        float norm();
        float angle();
        float distance(Vect2d v);

        float x;
        float y;
};
#endif // VECT2D_H
