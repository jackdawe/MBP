#include "mapss.h"

MapSS::MapSS(){}

MapSS::MapSS(vector<float> stateVector, int nWaypoints)
{
  for (int i=0;i<nWaypoints;i++)
    {
      Waypoint wp;
      wp.setCentre(Vect2d(stateVector[4+3*i], stateVector[5+3*i]));
      wp.setRadius(stateVector[6+i*3]);
      waypoints.push_back(wp);
    }
  int nPlanets = (stateVector.size() - 4-3*nWaypoints)/3;
  for (int i=0;i<nPlanets;i++)
    {
      Planet p;
      p.setCentre(Vect2d(stateVector[4+3*(i+nWaypoints)], stateVector[5+3*(i+nWaypoints)]));
      p.setRadius(stateVector[6+3*(i+nWaypoints)]);
      planets.push_back(p);      
    }
}

void MapSS::generate(int nPlanets, int planetMinSize, int planetMaxSize,int nWaypoints, int wpRadius)
{
  default_random_engine generator(random_device{}());
  uniform_int_distribution<int> dist;

  planets = vector<Planet>(); waypoints = vector<Waypoint>();
  
  //GENERATING THE PLANETS
  
  for (int i=0;i<nPlanets;i++)
    {
      Planet planet;
      dist = uniform_int_distribution<int>(planetMinSize,planetMaxSize);
      int radius = dist(generator);
      planet.setRadius(radius);
      dist = uniform_int_distribution<int>(radius,SIZE-radius);	
      planet.setCentre(Vect2d(dist(generator),dist(generator)));
      planets.push_back(planet);
    }
  
  //GENERATING WAYPOINTS
  
  dist = uniform_int_distribution<int>(wpRadius,SIZE-wpRadius);
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
}

void MapSS::generateMapPool(int nPlanets, int planetMinSize, int planetMaxSize,int nWaypoints, int wpRadius, string path, int nMaps)
{
  experimental::filesystem::create_directory(path);
  experimental::filesystem::create_directory(path+"train");
  experimental::filesystem::create_directory(path+"test");
  for (int i=0;i<nMaps;i++)
    {
      generate(nPlanets, planetMinSize, planetMaxSize, nWaypoints, wpRadius);
      save(path+"/train/map"+to_string(i));
      generate(nPlanets, planetMinSize, planetMaxSize, nWaypoints, wpRadius);
      save(path+"/test/map"+to_string(i));
    }
  
}

void MapSS::save(string filename)
{
  ofstream f(filename);
    if (f)
    {
        f << "--- PLANETS ---" << endl;
        for (unsigned int i=0;i<planets.size();i++)
        {
            f << to_string((int)planets[i].getRadius()) + " " + to_string((int)planets[i].getCentre().x) + " " + to_string((int)planets[i].getCentre().y)<<endl;
        }
        f << "--- WAYPOINTS ---" << endl;
        for (unsigned int i=0;i<waypoints.size();i++)
        {
	  f << to_string((int)waypoints[i].getRadius()) + " " + to_string((int)waypoints[i].getCentre().x) + " " + to_string((int)waypoints[i].getCentre().y)<<endl;
        }
	f <<"--- END ---" <<endl;
    }
    else
    {
        cout << "An error has occurred while trying to save the map" << endl;
    }
}

void MapSS::load(string filename)
{
  planets=vector<Planet>();
  waypoints=vector<Waypoint>();
  ifstream f(filename);
  if (f)
    {
        string line;
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
        while(line != "--- END ---")
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
    }
}

vector<Planet> MapSS::getPlanets() const
{
    return planets;
}

vector<Waypoint> MapSS::getWaypoints() const
{
    return waypoints;
}
