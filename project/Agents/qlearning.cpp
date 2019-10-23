#include "qlearning.h"

template <class C>
QLearning<C>::QLearning()
{
}

template <class C>
QLearning<C>::QLearning(C controller,int nEpisodes, float epsilon, float gamma):
    Agent<C>(controller), nEpisodes(nEpisodes),epsilon(epsilon), gamma(gamma)
{
    Agent<C>::generateNameTag("QL");
    //Initialising the Q fonction to 0 for each state action pair

    for (int i=0;i<controller.spaceStateSize();i++)
    {
        qvalues.push_back(vector<float>(this->actions().cardinal(),0));
    }
}

template <class C>
void QLearning<C>::epsilonGreedyPolicy()
{

    default_random_engine generator = default_random_engine(random_device{}());
    uniform_real_distribution<float> dist(0,1);
    if (dist(generator) < epsilon)
    {
        for (int i=0;i<this->daSize();i++)
        {
            this->controller.updateTakenAction(i,this->discreteActions()[i].pick());
        }
    }
    else
    {
        vector<float> possibleQValues;

        for (int i=0;i<Agent<C>::actions().cardinal();i++)
        {
            possibleQValues.push_back(qvalues[this->controller.stateId(this->currentState())][i]);
        }
        float maxQValue = *max_element(possibleQValues.begin(),possibleQValues.end());

        //If several qvalues are equal to the max, pick one of them randomly

        vector<int> indexOfMax;
        for (unsigned int i=0;i<possibleQValues.size();i++)
        {
            if (possibleQValues[i]==maxQValue)
            {
                indexOfMax.push_back(i);
            }
        }
        default_random_engine generator(random_device{}());
        uniform_int_distribution<int> dist(0,indexOfMax.size()-1);
        int actionId = indexOfMax[dist(generator)];
        this->controller.setTakenAction(this->actions().actionFromId(actionId,new vector<float>()));
    }
    this->controller.setTakenReward(this->controller.transition());
}

template <class C>
void QLearning<C>::updateQValues()
{
    int psIndex = this->controller.stateId(this->previousState());
    int csIndex = this->controller.stateId(this->currentState());
    int actionId = this->actions().idFromAction(this->takenAction());
    if (this->controller.isTerminal(this->previousState()))
    {
        for (int i=0;i<this->actions().cardinal();i++)
        {
            qvalues[psIndex][3-i] += (1./(this->episodeNumber+1))*(this->takenReward()+gamma*qvalues[psIndex][0]-qvalues[psIndex][0]);
        }
    }
    else
    {
        vector<float> updateChoice;
        for (int i=0;i<this->actions().cardinal();i++)
        {
            updateChoice.push_back(qvalues[csIndex][i]);
        }
        float bestChoice = *max_element(updateChoice.begin(),updateChoice.end());
        qvalues[psIndex][actionId] += (1./(sqrt(this->episodeNumber)+1))*(this->takenReward()+gamma*bestChoice-qvalues[psIndex][actionId]);
    }
}

template <class C>
void QLearning<C>::train()
{
    float e = epsilon;
    for (int k=0;k<nEpisodes;k++)
    {
        epsilon = e*exp(-k*5./nEpisodes);

        //Displaying a progression bar in the terminal

        if (nEpisodes > 100 && k%(5*nEpisodes/100) == 0)
        {
            cout << "Training in progress... " + to_string(k/(nEpisodes/100)) + "%" << endl;
        }
        bool terminal = false;
        while(!terminal)
        {
            epsilonGreedyPolicy();
            terminal = this->controller.isTerminal(this->currentState());
            updateQValues();
        }
        this->controller.transition();
        updateQValues();
        this->episodeNumber++;
        this->controller.reset();
    }
}

template<class C>
void QLearning<C>::playOne()
{
    bool terminal = false;
    while(!terminal)
    {
        epsilonGreedyPolicy();
        terminal = this->controller.isTerminal(this->currentState());
    }
    this->saveLastEpisode();
}

template <class C>
void QLearning<C>::saveQValues()
{
    ofstream f(this->controller.getPath()+"Policies/"+this->nameTag);
    if (f)
    {
        f << to_string(this->epsilon) << endl;
        f << to_string(gamma) << endl;
        for (unsigned int i=0;i<qvalues.size();i++)
        {
            for (unsigned int j=0;j<qvalues[0].size();j++)
            {
                f << to_string(qvalues[i][j]) + " ";
            }
            f << endl;
        }
    }
    else
    {
        cout << "An error has occured while trying to save the qvalues for qlearning" << endl;
    }
}

template <class C>
void QLearning<C>::loadQValues(string tag)
{
    this->nameTag = tag;
    ifstream f(this->controller.getPath()+"Policies/"+tag);
    if (f)
    {
        string line;
        getline(f,line);
        this->epsilon = stof(line);
        getline(f,line);
        gamma = stof(line);
        Agent<C>::generateNameTag("QL");
        for (unsigned int i=0;i<qvalues.size();i++)
        {
            getline(f,line);            
            int k=0;
            for (unsigned int j=0;j<qvalues[0].size();j++)
            {
                string num;
                while(line[k]!=' ')
                {
                    num+=line[k];
                    k++;
                }
                qvalues[i][j] = stof(num);
                k++;
            }
        }
    }
    else
    {
        cout << "An error has occured while trying to load the policy file" << endl;
    }
}


template <class C>
void QLearning<C>::saveTrainingData()
{

}

template class QLearning<ControllerGW>;
