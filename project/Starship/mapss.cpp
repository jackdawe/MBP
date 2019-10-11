#include "mapss.h"

MapSS::MapSS()
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

void MapSS::save()
{
    ofstream f("../Starship/Map" + mapTag + "/map");
    if (f)
    {
        f << to_string(size) << endl;
        f << "--- PLANETS ---" << endl;
        for (unsigned int i=0;i<planets.size();i++)
        {
            f << to_string(planets[i].getRadius()) + " " + to_string(planets[i].getCentre().getX())
                 + " " + to_string(planets[i].getCentre().getY())<<endl;
        }
        f << "--- WAYPOINTS ---" << endl;
        for (unsigned int i=0;i<waypoints.size();i++)
        {
            f << to_string(waypoints[i].getRadius()) + " " + to_string(waypoints[i].getCentre().getX())
                 + " " + to_string(waypoints[i].getCentre().getY())<<endl;
        }
        f << "--- SHIP ---" << endl;
        f << to_string(ship.getWidth()) + " " + to_string(ship.getHeight()) + " " + to_string(ship.getP().getX())
             + " " + to_string(ship.getP().getY())<<endl;
    }
    else
    {
        cout << "An error has occurred while trying to save the map" << endl;
    }
}

void MapSS::load(string mapTag)
{
    ifstream f("../Starship/Map" + mapTag + "/map");
    if (f)
    {
        string line;
        getline(f,line);
        size = stoi(line);
        getline(f,line);
        getline(f,line);
        unsigned int i=0;
        while(line != "--- WAYPOINTS ---")
        {
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
        i = 0;
        while(line != "--- SHIP ---")
        {
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
        i =0;
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

string MapSS::getMapTag() const
{
    return mapTag;
}
