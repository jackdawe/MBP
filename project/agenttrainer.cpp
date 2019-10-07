#include "agenttrainer.h"

template <class T>
AgentTrainer<T>::AgentTrainer()
{

}

template <class T>
void AgentTrainer<T>::train(T *agent,int numberOfEpisodes, int trainMode, int savingSequenceMode)
{
    for (int k=0;k<numberOfEpisodes;k++)
    {
        vector<vector<double>> actionSequence;
        double episodeTotalReward;
        State initialState;
        agent->setNextState(initialState);
        if (trainMode)
        {
            agent->initialiseEpisode();
        }
        bool terminal = false;
        while(!terminal)
        {
            agent->epsilonGreedyPolicy();
            terminal = agent->getNextState().isTerminal();
            actionSequence.push_back(agent->getTakenAction());
            episodeTotalReward+=agent->getTakenReward();
            if (trainMode)
            {
                agent->updatePolicy();
            }
        }
        if (savingSequenceMode)
        {
            saveEpisode(actionSequence,k);
        }
        agent->addToRewardHistory(episodeTotalReward);
    }
}

template <class T>
void AgentTrainer<T>::saveEpisode(vector<vector<double> > actionSequence, int seqId)
{
    ofstream f("../Sequences/seq" + seqId);
    if (f)
    {
        for (unsigned int i=0;i<actionSequence.size();i++)
        {
            for (unsigned int j=0;j<actionSequence[0].size();j++)

                f << to_string(actionSequence[i][j]) + " ";
            f << endl;
        }
    }
    else
    {
        cout<<"An error has occured when trying to save the sequence"<<endl;
    }
}
