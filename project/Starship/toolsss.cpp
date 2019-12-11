#include "toolsss.h"

ToolsSS::ToolsSS(){}

ToolsSS::ToolsSS(SpaceWorld sw): sw(sw){}

void ToolsSS::generateDataSet(string path, int nmaps, int n, int nTimesteps, float winProp)
{
  sw = SpaceWorld(path+"train/",nmaps);

  //Initialising the tensors that will contain the training set

  int size = sw.getSvSize();
  torch::Tensor stateInputs = torch::zeros({4*n/5,nTimesteps,size});
  torch::Tensor actionInputs = torch::zeros({4*n/5,nTimesteps,3});
  torch::Tensor stateLabels = torch::zeros({4*n/5,nTimesteps,size});
  torch::Tensor rewardLabels = torch::zeros({4*n/5,nTimesteps});

  //Making the agent wander randomly for n episodes 
  
  int j=0;
  for (int i=0;i<n;i++)
    {
      
      //Displaying a progression bar in the terminal
      
      if (n > 100 && i%(5*n/100) == 0)
	{
	  cout << "Your agent is crashing into planets for science... " + to_string(i/(n/100)) + "%" << endl;
	}

      //Swapping to test set generation when training set generation is done
      
      if (i==4*n/5)
	{
	  sw = SpaceWorld(path+"test/",nmaps);
	  j = 0;
	  cout<< "Training set generation is complete! Now generating test set..."<<endl; 
	  torch::save(stateInputs,path+"stateInputsTrain.pt");
	  torch::save(actionInputs,path+"actionInputsTrain.pt");
	  torch::save(rewardLabels,path+"rewardLabelsTrain.pt");
	  torch::save(stateLabels,path+"stateLabelsTrain.pt");
	  stateInputs = torch::zeros({n/5,nTimesteps,size});
	  actionInputs = torch::zeros({n/5,nTimesteps,3});
	  stateLabels = torch::zeros({n/5,nTimesteps,size});
	  rewardLabels = torch::zeros({n/5,nTimesteps});
	}

      for (int t=0;t<nTimesteps;t++)
	{
      
	  //Building the dataset tensors
      
	  stateInputs[j][t] = torch::tensor(sw.getCurrentState().getStateVector());
	  sw.setTakenAction(sw.randomAction());
	  actionInputs[j][t] = torch::tensor(sw.getTakenAction());
	  rewardLabels[j][t] = sw.transition();
	  stateLabels[j][t] = torch::tensor(sw.getCurrentState().getStateVector());
	}
      sw.reset();
      j++;

    }
      
  //Saving the test set
  
  cout<< "Test set generation is complete!"<<endl;
  torch::save(stateInputs,path+"stateInputsTest.pt");
  torch::save(actionInputs,path+"actionInputsTest.pt");
  torch::save(rewardLabels,path+"rewardLabelsTest.pt");
  torch::save(stateLabels,path+"stateLabelsTest.pt");  
}
