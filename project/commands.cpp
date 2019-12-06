#include "commands.h"
DEFINE_double(eps,0.1,"Probability of exploring for agents using epsilon greedy policies");
DEFINE_double(g,0.95,"Discount factor");
DEFINE_string(map,"../GridWorld/Maps/Inter8x8/train/map0","Path to a map file");
DEFINE_string(mp,"../GridWorld/Maps/Inter8x8/","Path to a directory containing your maps");

Commands::Commands(){}

void Commands::generateMapGW()
{
  MapGW map(FLAGS_size);
  map.generate(FLAGS_maxObst);
  map.save(FLAGS_f);
}

void Commands::generateMapPoolGW()
{
  MapGW map(FLAGS_size); 
  map.generateMapPool(FLAGS_maxObst,FLAGS_dir,FLAGS_nmaps);
}

void Commands::showMapGW(int argc, char* argv[])
{
  QApplication a(argc,argv);
  EpisodePlayerGW ep(FLAGS_f);
  ep.showMap();
  a.exec();
}

void Commands::trainQLAgentGW()
{
  GridWorld gw(FLAGS_map);
  gw.generateVectorStates();
  QLearning<GridWorld> agent(gw);
  agent.train(FLAGS_n,FLAGS_eps,FLAGS_g);
  agent.saveQValues("../temp/QValuesGW");
  agent.saveRewardHistory();
}

void Commands::playQLAgentGW(int argc, char* argv[])
{
  QApplication a(argc,argv);
  GridWorld gw(FLAGS_map);
  gw.generateVectorStates();
  QLearning<GridWorld> agent(gw);
  agent.loadQValues(FLAGS_f);
  agent.playOne();
  EpisodePlayerGW ep(FLAGS_map);
  ep.playEpisode(agent.getWorld().getStateSequence());
  a.exec();
}

void Commands::evaluateQLPolicy(int argc, char* argv[])
{
  vector<vector<string>> toDisplay;
  string filename = FLAGS_map;
  MapGW map;
  map.load(filename);
  int size = map.getSize();
  for (int i=0;i<size;i++)
    {
      vector<string> line;
      for (int j=0;j<size;j++)
	{
	  GridWorld gw(filename,i,j);
	  gw.generateVectorStates();
	  string sDirection; 
	  if (gw.isTerminal(gw.getCurrentState().getStateVector()))
	    {
	      sDirection = "";
	    }
	  else
	    {	  
	      
	      QLearning<GridWorld> agent(gw);
	      agent.loadQValues(FLAGS_f);    	  
	      agent.epsilonGreedyPolicy(0);	  
	      int iDirection = agent.takenAction()[0];
	      switch (iDirection)
		{
		case 0:
		  sDirection = "UP";
		  break;
		case 1:
		  sDirection = "RIGHT";
		  break;
		case 2:
		  sDirection = "DOWN";
		  break;
		case 3:
		  sDirection = "LEFT";
		  break;
		}
	    }
	  line.push_back(sDirection);
	}
      toDisplay.push_back(line);
    }
  QApplication a(argc,argv);  
  EpisodePlayerGW ep(FLAGS_map);
  ep.displayOnGrid(toDisplay);
  a.exec();  
}

void Commands::trainA2CGW()
{
  GridWorld gw(FLAGS_mp,FLAGS_nmaps);
  int size = gw.getSize();
  ConvNetGW net(size,FLAGS_conv1,FLAGS_conv2,FLAGS_fc1);
  ParametersA2C params(FLAGS_g, FLAGS_lr, FLAGS_beta, FLAGS_zeta, FLAGS_bs, FLAGS_n);
  ActorCritic<GridWorld,ConvNetGW> agent(gw,net);
  agent.train(FLAGS_n,FLAGS_g,FLAGS_beta,FLAGS_zeta,FLAGS_lr,FLAGS_bs);
  torch::save(agent.getModel(),"../temp/CNN_A2C_GW.pt");
}

/*

void Commands::testA2C()
{
  ConvNetGW net(8,16,16,128);
  torch::load(net,"../model.pt");
  float tot=0;
  for (int i=0;i<FLAGS_nmaps;i++)
    {
      MapGW map;
      map.load(FLAGS_dir+"map"+to_string(i));
      int size = map.getSize();
      vector<float> mapRewards;
      int emptySpaces=0;
      int accuracy = 0;
      for (int x=0;x<size;x++)
	{
	  for (int y=0;y<size;y++)
	    {
	      if (map.getMap()[x][y] == 0)
		{
		  emptySpaces++;
		  GridWorld gw(FLAGS_dir+"map"+to_string(i),x,y);
		  gw.generateVectorStates();
		  ActorCritic<GridWorld,ConvNetGW> agent(gw,net,true);
		  agent.playOne();
		  if (agent.takenReward()>0)
		    {
		      accuracy++;
		    }
		}
	    }
	}
      accuracy=accuracy*100./emptySpaces;
      tot+=accuracy;
      cout << "The model completed a " + to_string(accuracy) + "% accuracy on map" + to_string(i) << endl
	;
    }
  cout<<"The overall accuracy on the test set: " + to_string(tot/FLAGS_nmaps) +"%"<<endl;
}
	       
void Commands::showCriticOnMapGW(int argc, char* argv[])
{
  QApplication a(argc,argv);
  vector<vector<string>> toDisplay;
  string filename = FLAGS_f;
  MapGW map;
  map.load(filename);
  int size = map.getSize();
  ConvNetGW net(8,16,16,128);
  torch::load(net,"../model.pt");
  for (int i=0;i<size;i++)
    {
      vector<string> line;
      for (int j=0;j<size;j++)
	{
	  GridWorld gw(filename,i,j);
	  gw.generateVectorStates();
	  torch::Tensor output = net->criticOutput(gw.toRGBTensor(gw.getCurrentState().getStateVector()).to(net->getUsedDevice()));
	  string val = to_string(*output.to(torch::Device(torch::kCPU)).data<float>());
	  string val2;
	  val2+=val[0],val2+=val[1],val2+=val[2],val2+=val[3],val2+=val[4];
	  line.push_back(val2);
	}
      toDisplay.push_back(line);
    }
  EpisodePlayerGW ep(FLAGS_f);
  ep.displayOnGrid(toDisplay);
  a.exec(); 
}

void Commands::showActorOnMapGW(int argc, char* argv[])
{
  QApplication a(argc,argv);
  vector<vector<string>> toDisplay;
  string filename = FLAGS_f;
  MapGW map;
  map.load(filename);
  int size = map.getSize();
  ConvNetGW net(8,16,16,128);
  torch::load(net,"../model.pt");
  for (int i=0;i<size;i++)
    {
      vector<string> line;
      for (int j=0;j<size;j++)
	{
	  GridWorld gw(filename,i,j);
	  gw.generateVectorStates();
	  gw.toRGBTensor(gw.getCurrentState().getStateVector());
	  torch::Tensor output = net->actorOutput(gw.toRGBTensor(gw.getCurrentState().getStateVector()).to(net->getUsedDevice()));
	  float max =0;
	  int iDirection;
	  string sDirection;
	  for (int k=0;k<4;k++)
	    {
	      float val = *output[0][k].to(torch::Device(torch::kCPU)).data<float>();
	      if (val>max)
                {
		  max = val;
		  iDirection = k;
                }
            }
	  switch (iDirection)
            {
            case 0:
	      sDirection = "UP";
	      break;
            case 1:
	      sDirection = "RIGHT";
	      break;
            case 2:
	      sDirection = "DOWN";
	      break;
            case 3:
	      sDirection = "LEFT";
	      break;
            }
	  line.push_back(sDirection);
	}
      toDisplay.push_back(line);
    }
  EpisodePlayerGW ep(FLAGS_f);
  ep.displayOnGrid(toDisplay);
  a.exec();
}
*/
void Commands::generateDataSetGW()
{
  ToolsGW t;
  t.generateDataSet(FLAGS_dir,FLAGS_nmaps,FLAGS_n,FLAGS_wp);
}

void Commands::learnTransitionFunctionGW()
{
  GridWorld gw;
  ToolsGW t(gw);
  string path = FLAGS_dir;
  torch::Tensor stateInputsTr, actionInputsTr, stateLabelsTr;
  torch::Tensor stateInputsTe, actionInputsTe, stateLabelsTe;
  torch::load(stateInputsTr,path+"stateInputsTrain.pt");
  torch::load(actionInputsTr, path+"actionInputsTrain.pt");
  torch::load(stateLabelsTr, path+"stateLabelsTrain.pt");
  torch::load(stateInputsTe,path+"stateInputsTest.pt");
  torch::load(actionInputsTe, path+"actionInputsTest.pt");
  torch::load(stateLabelsTe, path+"stateLabelsTest.pt");

  if (FLAGS_wn)
    {
      actionInputsTe+=torch::zeros({actionInputsTe.size(0),actionInputsTe.size(1)}).normal_(0,FLAGS_sd);
      actionInputsTr+=torch::zeros({actionInputsTr.size(0),actionInputsTr.size(1)}).normal_(0,FLAGS_sd);
    }
  
  TransitionGW ft(stateInputsTr.size(2),FLAGS_sc1,FLAGS_afc1,FLAGS_afc2);
  ft->to(torch::Device(torch::kCUDA));
  ModelBased<GridWorld,TransitionGW,RewardGW, PlannerGW> agent(gw,ft);
  agent.learnTransitionFunction(actionInputsTr, stateInputsTr, stateLabelsTr,FLAGS_n,FLAGS_bs,FLAGS_lr);
  agent.saveTrainingData();
  torch::save(agent.getTransitionFunction(),"../temp/TransitionGW.pt");
  agent.getTransitionFunction()->saveParams("../temp/TransitionGW_Params");
  
  //Computing accuracy

  {
    torch::NoGradGuard no_grad;
    auto model = agent.getTransitionFunction();
    t.transitionAccuracy(model->predictState(stateInputsTe.to(model->getUsedDevice()),actionInputsTe.to(model->getUsedDevice())),stateLabelsTe);
  } 
}

void Commands::learnRewardFunctionGW()
{
  GridWorld gw;
  ToolsGW t(gw);
  string path = FLAGS_dir;
  torch::Tensor stateInputsTr, actionInputsTr, rewardLabelsTr;
  torch::Tensor stateInputsTe, actionInputsTe, rewardLabelsTe;
  torch::load(stateInputsTr,path+"stateInputsTrain.pt");
  torch::load(actionInputsTr, path+"actionInputsTrain.pt");
  torch::load(rewardLabelsTr, path+"rewardLabelsTrain.pt");
  torch::load(stateInputsTe,path+"stateInputsTest.pt");
  torch::load(actionInputsTe, path+"actionInputsTest.pt");
  torch::load(rewardLabelsTe, path+"rewardLabelsTest.pt");

  if (FLAGS_wn)
    {
      actionInputsTe+=torch::zeros({actionInputsTe.size(0),actionInputsTe.size(1)}).normal_(0,FLAGS_sd);
      actionInputsTr+=torch::zeros({actionInputsTr.size(0),actionInputsTr.size(1)}).normal_(0,FLAGS_sd);
    }
  RewardGW fr(stateInputsTr.size(3),FLAGS_sc1,FLAGS_afc1,FLAGS_afc2);
  fr->to(torch::Device(torch::kCUDA));
  ModelBased<GridWorld,TransitionGW,RewardGW, PlannerGW> agent(gw,fr);
  agent.learnRewardFunction(actionInputsTr, stateInputsTr, rewardLabelsTr,FLAGS_n,FLAGS_bs,FLAGS_lr);
  agent.saveTrainingData();
  torch::save(agent.getRewardFunction(),"../temp/RewardGW.pt");
  agent.getRewardFunction()->saveParams("../temp/RewardGW_Params");

  //Computing accuracy

  {
    torch::NoGradGuard no_grad;
    auto model = agent.getRewardFunction();
    t.rewardAccuracy(model->predictReward(stateInputsTe.to(model->getUsedDevice()),actionInputsTe.to(model->getUsedDevice())),rewardLabelsTe);
  }


}

void Commands::test()
{
  TransitionGW ft("../temp/TransitionGW_Params");
  torch::load(ft,"../temp/TransitionGW.pt");  
  RewardGW fr("../temp/RewardGW_Params");
  torch::load(fr,"../temp/RewardGW.pt");
  GridWorld gw("../GridWorld/Maps/Inter8x8/train/map3",3,5);
  gw.generateVectorStates();
  ModelBased<GridWorld,TransitionGW,RewardGW,PlannerGW> agent(gw,ft,fr,PlannerGW());
  agent.gradientBasedPlanner(FLAGS_K,FLAGS_T,FLAGS_gs,FLAGS_lr);

  /*
  ofstream f("../hello");
  for (int i=0;i<10000;i++)
    {
      torch::Tensor a = torch::tensor({1-(i/10000.),i/10000.,0.,0.}).to(torch::kFloat32);
      torch::Tensor s = torch::tensor(gw.getCurrentState().getStateVector());
      torch::Tensor r = fr->predictReward(s.unsqueeze(0),a.unsqueeze(0).to(fr->getUsedDevice()))[0].to(torch::Device(torch::kCPU));
      f<<*r.data<float>()<<endl;
    }
  */
}

void Commands::learnForwardModelGW()
{
  GridWorld gw;
  ToolsGW t(gw);
  string path = FLAGS_dir;
  torch::Tensor stateInputsTr, actionInputsTr, stateLabelsTr, rewardLabelsTr;
  torch::Tensor stateInputsTe, actionInputsTe, stateLabelsTe, rewardLabelsTe;
  torch::load(stateInputsTr,path+"stateInputsTrain.pt");
  torch::load(actionInputsTr, path+"actionInputsTrain.pt");
  torch::load(stateLabelsTr,path+"stateLabelsTrain.pt");
  torch::load(rewardLabelsTr, path+"rewardLabelsTrain.pt");
  torch::load(stateInputsTe,path+"stateInputsTest.pt");
  torch::load(actionInputsTe, path+"actionInputsTest.pt");
  torch::load(stateLabelsTe,path+"stateLabelsTest.pt");
  torch::load(rewardLabelsTe, path+"rewardLabelsTest.pt");

  if (FLAGS_wn)
    {
      actionInputsTr+=torch::zeros({actionInputsTr.size(0),actionInputsTr.size(1)}).normal_(0,FLAGS_sd);
    }
  ForwardGW forwardModel(stateInputsTr.size(3),FLAGS_sc1);
  forwardModel->to(torch::Device(torch::kCUDA));
  ModelBased2<GridWorld,ForwardGW, PlannerGW> agent(gw,forwardModel);
  agent.learnForwardModel(actionInputsTr, stateInputsTr,stateLabelsTr, rewardLabelsTr,FLAGS_n,FLAGS_bs,FLAGS_lr);
  agent.saveTrainingData();
  torch::save(agent.getForwardModel(),"../temp/ForwardGW.pt");
  agent.getForwardModel()->saveParams("../temp/ForwardGW_Params");
  //Computing accuracy

  {
    torch::NoGradGuard no_grad;
    auto model = agent.getForwardModel();
    model->forward(stateInputsTe.to(model->getUsedDevice()),actionInputsTe.to(model->getUsedDevice()));
    t.rewardAccuracy(model->predictedReward.to(torch::Device(torch::kCPU)),rewardLabelsTe); 
    t.transitionAccuracy(model->predictedState.to(torch::Device(torch::kCPU)),stateLabelsTe);
    
  }


}

void Commands::test2()
{
  ForwardGW fm("../temp/ForwardGW_Params");
  torch::load(fm,"../temp/ForwardGW.pt");
  GridWorld gw("../GridWorld/Maps/Inter8x8/train/map1",6,6);
  gw.generateVectorStates();
  ModelBased2<GridWorld,ForwardGW,PlannerGW> agent(gw,fm,PlannerGW());
  agent.gradientBasedPlanner(FLAGS_K,FLAGS_T,FLAGS_gs,FLAGS_lr);

  /*
  ofstream f("../hello4");
  for (int i=0;i<10000;i++)
    {
      torch::Tensor a = torch::tensor({1-(i/10000.),i/10000.,0.,0.}).to(torch::kFloat32);
      torch::Tensor s = torch::tensor(gw.getCurrentState().getStateVector());
      fm->forward(s.unsqueeze(0),a.unsqueeze(0).to(fm->getUsedDevice()));
      f<<*fm->predictedReward.to(torch::Device(torch::kCPU)).data<float>()<<endl;
    }
  */
}

void Commands::generateMapSS()
{
  MapSS map(1000);
  map.generate();
  map.save(FLAGS_map);
}

void Commands::showMapSS(int argc, char* argv[])
{
  QApplication a(argc,argv);
  EpisodePlayerSS ep(FLAGS_map);
  ep.showMap();
  a.exec();
}
