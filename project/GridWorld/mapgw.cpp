#include "mapgw.h"


MapGW::MapGW()
{
}

MapGW::MapGW(int size, string mapTag): size(size), mapTag(mapTag)
{
    //Initialising the map matrix with empty spaces

    for (int i=0;i<size;i++)
    {
        map.push_back(vector<int>(size,0));
    }
}

void MapGW::generate(int obstacleMaxNumber)
{

    default_random_engine generator(random_device{}());
    uniform_int_distribution<int> dist(1,size-2);

    //Setting the walls

    for (int i=0;i<size;i++)
    {
        map[0][i]=1;
        map[i][0]=1;
        map[size-1][i]=1;
        map[i][size-1]=1;
    }

    //Setting the agent's objective

    int i = dist(generator);
    int j = dist(generator);
    map[i][j] = 2;

    //Setting the obstacles

    for (int k=0;k<obstacleMaxNumber;k++)
    {
        i = dist(generator);
        j = dist(generator);
        while(map[i][j]==3 || map[i][j]==2)
        {
            i = dist(generator);
            j = dist(generator);
        }
        map[i][j] = 1;
    }
}

void MapGW::save()
{
    ofstream f("../GridWorld/Map" + mapTag + "/map");
    if (f)
    {
        f << size << endl;
        for (int i=0;i<size;i++)
        {
            for (int j=0;j<size-1;j++)
            {
                f<< std::to_string(map[i][j]) + " ";
            }
            f<<std::to_string(map[i][size-1])<<endl;
        }
    }
    else
    {
        cout << "An error has occured while trying to save the map file" << endl;
    }


}

void MapGW::load(string mapTag)
{
    this->mapTag=mapTag;
    ifstream f("../GridWorld/Map" + mapTag + "/map");
    string line;
    int i=0;
    getline(f,line);
    size = stoi(line);
    for (int i=0;i<size;i++)
    {
        map.push_back(vector<int>(size,0));
    }
    while (std::getline(f,line))
    {
        for (int j=0;j<size;j++)
        {
            map[i][j] = line[2*j] - '0';
        }
        i++;
    }
}

int MapGW::getSize() const
{
    return size;
}

string MapGW::getMapTag() const
{
    return mapTag;
}

vector<vector<int> > MapGW::getMap() const
{
    return map;
}
