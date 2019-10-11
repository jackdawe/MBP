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

        vector<vector<float>> stateSequence;
        float episodeTotalReward;
        agent->initialiseEpisode();
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
void AgentTrainer<A>::saveEpisode(vector<vector<float> > stateSequence, int seqId)
{
    ofstream f("../Sequences/seq" + to_string(seqId));
    if (f)
    {
        for (unsigned int i=0;i<stateSequence.size();i++)
        {
            for (unsigned int j=0;j<stateSequence[0].size();j++)

                f << to_string((int)stateSequence[i][j]) + " ";
            f << endl;
        }
    }
    else
    {
        cout<<"An error has occured when trying to save the sequence"<<endl;
    }
}

template <class A>
vector<vector<float>> AgentTrainer<A>::loadEpisode(int seqId)
{
    vector<vector<float>> sequence;
    ifstream f("../Sequences/seq" + to_string(seqId));
    if (f)
    {
        string line;
        while(getline(f,line))
        {
            int i=0;
            vector<float> vecline;
            while(i<line.size())
            {
                vecline.push_back(line[i]-48);
                i+=2;
            }
            sequence.push_back(vecline);
        }
    }
    else
    {
        cout<<"An error has occured when trying to load the sequence"<<endl;
    }
    return sequence;
}

template class AgentTrainer<QLearning<ControllerGW>>;
