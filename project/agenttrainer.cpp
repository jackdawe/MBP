#include "agenttrainer.h"

template <class A>
AgentTrainer<A>::AgentTrainer()
{

}

template <class A>
void AgentTrainer<A>::train(A *agent,int numberOfEpisodes, int trainMode, int savingSequenceMode)
{
    for (int k=0;k<numberOfEpisodes;k++)
    {
        //Displaying a progression bar in the terminal

        if (numberOfEpisodes > 100 && k%(5*numberOfEpisodes/100) == 0)
        {
            cout << "Training in progress... " + to_string(k/(numberOfEpisodes/100)) + "%" << endl;
        }
        float episodeTotalReward;
        agent->initialiseEpisode();
        bool terminal = false;
        int i = 0;
        while(!terminal && i != 1000)
        {            
            i++;
            agent->epsilonGreedyPolicy();
            terminal = agent->getController().isTerminal(agent->currentState());
            if(savingSequenceMode)
            {
                actionSequence.push_back(agent->takenAction());
                stateSequence.push_back(agent->currentState().getStateVector());
            }
            episodeTotalReward+=agent->takenReward();
            if (trainMode)
            {
                agent->updatePolicy();
            }
        }
        if (savingSequenceMode)
        {
            saveEpisode(*agent,k);
        }
        agent->addToRewardHistory(episodeTotalReward);
        agent->getController().reset();
    }
}

template <class A>
void AgentTrainer<A>::saveEpisode(A agent, int seqId)
{
    ofstream f("../Sequences/seq" + to_string(seqId));
    if (f)
    {
        vector<string> paramLabels = agent.getController().getParamLabels();
        vector<float> paramValues = agent.getController().getParamValues();
        for (unsigned int i=0; i<paramLabels.size();i++)
        {
            f << paramLabels[i] + " = " + to_string(paramValues[i])<<endl;;
        }
        f << "---SEQUENCE---" <<endl;
        for (unsigned int i=0;i<stateSequence.size();i++)
        {
            string line;
            for (unsigned int j=0;j<actionSequence[0].size();j++)
            {
                line += to_string(actionSequence[i][j]) + " ";
            }
            line += "| ";
            for (unsigned int j=0;j<stateSequence[0].size();j++)

                line+= to_string(stateSequence[i][j]) + " ";
            f << line << endl;
        }
    }
    else
    {
        cout<<"An error has occured when trying to save the sequence"<<endl;
    }
}

template <class A>
void AgentTrainer<A>::loadSequence(int seqId)
{
    stateSequence = vector<vector<float>>();
    actionSequence = vector<vector<float>>();
    ifstream f("../Sequences/seq" + to_string(seqId));
    if (f)
    {
        string line;
        while (line != "---SEQUENCE---")
        {
            getline(f,line);
        }
        while(getline(f,line))
        {
            int a = line.size();
            int i=0;
            vector<float> vecline;          
            while (line[i] != '|')
            {
                string num;
                while(line[i] != ' ')
                {
                    num+=line[i];
                    i++;
                }
                vecline.push_back(stof(num));
                i++;
            }
            i+=2;
            actionSequence.push_back(vecline);
            vecline = vector<float>();
            while (i < line.size())
            {
                string num;
                while(line[i] != ' ')
                {
                    num+=line[i];
                    i++;
                }
                vecline.push_back(stof(num));
                i++;
            }
            stateSequence.push_back(vecline);
        }
    }
    else
    {
        cout<<"An error has occured when trying to load the sequence"<<endl;
    }
}

template <class A>
vector<float> AgentTrainer<A>::loadParameters(int seqId)
{
    vector<float> paramValues;
    ifstream f("../Sequences/seq" + to_string(seqId));
    if (f)
    {
        string line;
        getline(f,line);
        while (line != "---SEQUENCE---")
        {
            string param;
            int i=0;
            while (line[i] != '=')
            {
                i++;
            }
            i+=2;
            while (i != line.size())
            {
                param += line[i];
                i++;
            }
            paramValues.push_back(stof(param));
            getline(f,line);
        }
    }
    else
    {
        cout<<"An error has occured when trying to load the parameters used for the sequence"<<endl;
    }
    return paramValues;
}

template<class A>
vector<vector<float> > AgentTrainer<A>::getActionSequence() const
{
    return actionSequence;
}

template<class A>
vector<float> AgentTrainer<A>::getParameters() const
{
    return parameters;
}

template<class A>
vector<vector<float> > AgentTrainer<A>::getStateSequence() const
{
    return stateSequence;
}

template class AgentTrainer<QLearning<ControllerGW>>;
template class AgentTrainer<RandomAgent<ControllerSS>>;
