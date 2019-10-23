#include "agenttrainer.h"

template <class A>
AgentTrainer<A>::AgentTrainer()
{

}

template <class A>
void AgentTrainer<A>::train(A *agent,int numberOfEpisodes, int trainMode, int savingSequenceMode)
{
//    float e = agent->epsilon;
//    for (int k=0;k<numberOfEpisodes;k++)
//    {
//        agent->epsilon = e*exp(-k*5./numberOfEpisodes);

//        //Displaying a progression bar in the terminal

//        if (numberOfEpisodes > 100 && k%(5*numberOfEpisodes/100) == 0)
//        {
//            cout << "Training in progress... " + to_string(k/(numberOfEpisodes/100)) + "%" << endl;
//        }
//        agent->initialiseEpisode();
//        bool terminal = false;
//        while(!terminal)
//        {
//            agent->epsilonGreedyPolicy();
//            terminal = agent->getController().isTerminal(agent->currentState());
//            if (trainMode)
//            {
//                agent->updatePolicy();
//            }
//        }
//        agent->finaliseEpisode();
//        if (savingSequenceMode)
//        {
//            agent->saveLastEpisode();
//        }
//        agent->incrementEpisode();
//        agent->resetController();
//    }
}

//template <class A>
//void AgentTrainer<A>::displayScoresGW(string mapTag, string policyTag)
//{
//    MapGW map;
//    map.load(mapTag);
//    vector<vector<int>> r;
//    for (int i=0;i<map.getSize();i++)
//    {
//        vector<int> rl;
//        for (int j=0;j<map.getSize();j++)
//        {
//            ControllerGW controller(mapTag,i,j);
//            QLearning<ControllerGW> agent(controller,0.25,0.9);
//            agent.loadPolicy(policyTag);
//            agent.epsilon = 0;
//            this->train(&agent,1,0,0);
//            rl.push_back((int)agent.getController().getRewardHistory().front());
//        }
//        r.push_back(rl);
//    }
//    EpisodePlayerGW ep(mapTag);
//    ep.showScores(r);
//}

//template <class A>
//vector<float> AgentTrainer<A>::loadParameters(int seqId)
//{
//    vector<float> paramValues;
//    ifstream f("../Sequences/seq" + to_string(seqId));
//    if (f)
//    {
//        string line;
//        getline(f,line);
//        while (line != "---SEQUENCE---")
//        {
//            string param;
//            int i=0;
//            while (line[i] != '=')
//            {
//                i++;
//            }
//            i+=2;
//            while (i != line.size())
//            {
//                param += line[i];
//                i++;
//            }
//            paramValues.push_back(stof(param));
//            getline(f,line);
//        }
//    }
//    else
//    {
//        cout<<"An error has occured when trying to load the parameters used for the sequence"<<endl;
//    }
//    return paramValues;
//}

template class AgentTrainer<QLearning<ControllerGW>>;
