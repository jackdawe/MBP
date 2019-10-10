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
        vector<vector<double>> stateSequence;
        double episodeTotalReward;
        if (trainMode)
        {
            agent->initialiseEpisode();
        }
        bool terminal = false;
        while(!terminal)
        {
            stateSequence.push_back(agent->currentState().getStateVector());
            agent->epsilonGreedyPolicy();
            terminal = agent->getController().isTerminal(agent->currentState());
            episodeTotalReward+=agent->takenReward();
            if (trainMode)
            {
                agent->updatePolicy();
            }
        }
        if (savingSequenceMode)
        {
            saveEpisode(stateSequence,k);
        }
        agent->addToRewardHistory(episodeTotalReward);
    }
}

template <class A>
void AgentTrainer<A>::saveEpisode(vector<vector<double> > stateSequence, int seqId)
{
    ofstream f("../Sequences/seq" + seqId);
    if (f)
    {
        for (unsigned int i=0;i<stateSequence.size();i++)
        {
            for (unsigned int j=0;j<stateSequence[0].size();j++)

                f << to_string(stateSequence[i][j]) + " ";
            f << endl;
        }
    }
    else
    {
        cout<<"An error has occured when trying to save the sequence"<<endl;
    }
}

template class AgentTrainer<QLearning<ControllerGW>>;
