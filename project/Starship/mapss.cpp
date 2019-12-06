#include "mapss.h"

MapSS::MapSS()
{

}

MapSS::MapSS(int size): size(size)
{
}

void MapSS::generate(int nPlanets, int planetMinSize,
                     int planetMaxSize,int nWaypoints,
                     int wpRadius,int shipW, int shipH)
{
    default_random_engine generator(random_device{}());
    uniform_int_distribution<int> dist;

    //GENERATING THE PLANETS

    for (int i=0;i<nPlanets;i++)
    {
        Planet planet;
        dist = uniform_int_distribution<int>(planetMinSize,planetMaxSize);
        int radius = dist(generator);
        planet.setRadius(radius);
        dist = uniform_int_distribution<int>(radius,size-radius);
        planet.setCentre(Vect2d(dist(generator),dist(generator)));
        planets.push_back(planet);
    }

    //GENERATING WAYPOINTS

    dist = uniform_int_distribution<int>(wpRadius,size-wpRadius);
    for (int k=0;k<nWaypoints;k++)
    {
        Waypoint waypoint;
        waypoint.setRadius(wpRadius);
        bool invalidPosition = true;
        while (invalidPosition)
        {
            invalidPosition = false;
            Vect2d spawn(dist(generator),dist(generator));
            waypoint.setCentre(spawn);
            for (int i=0;i<nPlanets;i++)
            {
                if (spawn.distance(planets[i].getCentre()) < 1.1 * (wpRadius+planets[i].getRadius())) {
                    invalidPosition = true;
                    break;
                }
            }
            if (!invalidPosition)
            {
                for (unsigned int i=0;i<waypoints.size();i++)
                {
                    if (spawn.distance(waypoints[i].getCentre()) < 2.2 * wpRadius)
                    {
                        invalidPosition = true;
                        break;
                    }
                }
            }
        }
        waypoints.push_back(waypoint);
    }

    //GENERATING SHIP

    dist = uniform_int_distribution<int>(shipH,size-shipH);
    ship.setWidth(shipW);
    ship.setHeight(shipH);
    bool invalidPosition = true;
    while (invalidPosition)
    {
        invalidPosition = false;
        Vect2d spawn(dist(generator),dist(generator));
        ship.setP(spawn);
        for (int i=0;i<nPlanets;i++)
        {
            if (spawn.distance(planets[i].getCentre()) < 1.1 * (shipH+planets[i].getRadius())) {
                invalidPosition = true;
                break;
            }
        }
        if (!invalidPosition)
        {
            for (int i=0;i<nWaypoints;i++)
            {
                if (spawn.distance(waypoints[i].getCentre()) < 1.1 * (wpRadius + shipH))
                {
                    invalidPosition = true;
                    break;
                }
            }
        }
    }
}

void MapSS::save(string filename)
{
  ofstream f(filename);
    if (f)
    {
        f << to_string(size) << endl;
        f << "--- PLANETS ---" << endl;
        for (unsigned int i=0;i<planets.size();i++)
        {
            f << to_string((int)planets[i].getRadius()) + " " + to_string((int)planets[i].getCentre().getX())
                 + " " + to_string((int)planets[i].getCentre().getY())<<endl;
        }
        f << "--- WAYPOINTS ---" << endl;
        for (unsigned int i=0;i<waypoints.size();i++)
        {
            f << to_string((int)waypoints[i].getRadius()) + " " + to_string((int)waypoints[i].getCentre().getX())
                 + " " + to_string((int)waypoints[i].getCentre().getY())<<endl;
        }
        f << "--- SHIP ---" << endl;
        f << to_string(ship.getWidth()) + " " + to_string(ship.getHeight()) + " " + to_string((int)ship.getP().getX())
             + " " + to_string((int)ship.getP().getY())<<endl;
    }
    else
    {
        cout << "An error has occurred while trying to save the map" << endl;
    }
}

void MapSS::load(string filename)
{
  ifstream f(filename);
    if (f)
    {
        string line;
        getline(f,line);
        size = stoi(line);
        getline(f,line);
        getline(f,line);

        while(line != "--- WAYPOINTS ---")
        {
            unsigned int i=0;
            Planet p;
            string num;
            while(line[i]!=' ')
            {
                num+=line[i];
                i++;
            }
            i++;
            p.setRadius(stoi(num));
            num = "";
            while(line[i]!=' ')
            {
                num+=line[i];
                i++;
            }
            i++;
            int x = stoi(num);
            num = "";
            while(i != line.size())
            {
                num+=line[i];
                i++;
            }
            int y = stoi(num);
            p.setCentre(Vect2d(x,y));
            planets.push_back(p);
            getline(f,line);
        }
        getline(f,line);
        while(line != "--- SHIP ---")
        {
            unsigned int i=0;
            Waypoint wp;
            string num;
            while(line[i]!=' ')
            {
                num+=line[i];
                i++;
            }
            i++;
            wp.setRadius(stoi(num));
            num = "";
            while(line[i]!=' ')
            {
                num+=line[i];
                i++;
            }
            i++;
            int x = stoi(num);
            num = "";
            while(i != line.size())
            {
                num+=line[i];
                i++;
            }
            int y = stoi(num);
            wp.setCentre(Vect2d(x,y));
            waypoints.push_back(wp);
            getline(f,line);
        }
        getline(f,line);
        unsigned int i=0;
        string num;
        while(line[i]!=' ')
        {
            num+=line[i];
            i++;
        }
        i++;
        ship.setWidth(stoi(num));
        num = "";
        while(line[i]!=' ')
        {
            num+=line[i];
            i++;
        }
        i++;
        ship.setHeight(stoi(num));
        num = "";
        while(line[i]!=' ')
        {
            num+=line[i];
            i++;
        }
        i++;
        int x = stoi(num);
        num = "";
        while(i != line.size())
        {
            num+=line[i];
            i++;
        }
        int y = stoi(num);
        ship.setP(Vect2d(x,y));
        getline(f,line);
    }
}

int MapSS::getSize() const
{
    return size;
}

vector<Planet> MapSS::getPlanets() const
{
    return planets;
}

vector<Waypoint> MapSS::getWaypoints() const
{
    return waypoints;
}

Ship MapSS::getShip() const
{
    return ship;
}
