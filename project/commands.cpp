#include "commands.h"
DEFINE_double(eps,0.1,"Probability of exploring for agents using epsilon greedy policies");
DEFINE_double(g,0.95,"Discount factor");
DEFINE_string(map,"../GridWorld/Maps/Inter8x8/train/map0","Path to a map file");
DEFINE_string(mp,"../GridWorld/Maps/Inter8x8/","Path to a directory containing your maps");
DEFINE_int32(K,1,"Number of rollouts");
DEFINE_int32(T,1,"Number of timesteps to unroll");
DEFINE_int32(gs,1,"Number of gradient steps");
DEFINE_int32(sc1,16,"Number of feature maps of the first conv layer of the encoder. Next layers have twice as many features maps and the NN is shaped accordingly");

DEFINE_bool(asp,true,"If true, all input states are provided for training for model based agent. If false, only initial state and the action sequence are provided and the agent uses his predicted states to predict the next state"); 

DEFINE_string(mdl,"../temp/model","Path to a model file. Do not add the .pt extension.");
DEFINE_string(tag,"","Suffix for auto generated files"); 

//Starship flags

DEFINE_int32(nplan,N_PLANETS,"Number of planets for mapss generation");
DEFINE_int32(pmin,PLANET_MIN_SIZE,"Planet minimum radius for mapss generation");
DEFINE_int32(pmax,PLANET_MAX_SIZE,"Planet maximum radius for mapss generation");
DEFINE_int32(nwp,N_WAYPOINTS,"Number of waypoints for mapss generation");
DEFINE_int32(rwp,WAYPOINT_RADIUS,"Waypoint radius for mapss generation");
DEFINE_double(esp,0.1,"Proportion of episodes for which the agent will spawn near the edge of the map");
DEFINE_double(trp,0.9,"Share of the training set from the whole dataset");

Commands::Commands(){}

void Commands::generateMapGW()
{
  MapGW map(FLAGS_size);
  map.generate(FLAGS_maxObst);
  map.save(FLAGS_map);
}

void Commands::generateMapPoolGW()
{
  MapGW map(FLAGS_size); 
  map.generateMapPool(FLAGS_maxObst,FLAGS_mp,FLAGS_nmaps);
}

void Commands::showMapGW(int argc, char* argv[])
{
  QApplication a(argc,argv);
  EpisodePlayerGW ep(FLAGS_map);
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
  ActorCritic<GridWorld,ConvNetGW> agent(gw,net);
  agent.train(FLAGS_n,FLAGS_g,FLAGS_beta,FLAGS_zeta,FLAGS_lr,FLAGS_bs);
  torch::save(agent.getModel(),"../temp/CNN_A2C_GW.pt");
}



void Commands::testA2C()
{
  ConvNetGW net(8,32,32,128);
  torch::load(net,"../temp/CNN_A2C_GW.pt");
  float tot=0;
  for (int i=0;i<FLAGS_nmaps;i++)
    {
      MapGW map;
      map.load(FLAGS_mp+"map"+to_string(i));
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
		  GridWorld gw(FLAGS_mp+"map"+to_string(i),x,y);
		  gw.generateVectorStates();
		  ActorCritic<GridWorld,ConvNetGW> agent(gw,net);
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
  string filename = FLAGS_map;
  MapGW map;
  map.load(filename);
  int size = map.getSize();
  ConvNetGW net(8,32,32,128);
  torch::load(net,"../temp/CNN_A2C_GW.pt");
  for (int i=0;i<size;i++)
    {
      vector<string> line;
      for (int j=0;j<size;j++)
	{
	  GridWorld gw(filename,i,j);
	  gw.generateVectorStates();
	  torch::Tensor output = net->criticOutput(torch::tensor(gw.getCurrentState().getStateVector())).to(net->getUsedDevice());
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
  string filename = FLAGS_map;
  MapGW map;
  map.load(filename);
  int size = map.getSize();
  ConvNetGW net(8,32,32,128);
  torch::load(net,"../temp/CNN_A2C_GW.pt");
  for (int i=0;i<size;i++)
    {
      vector<string> line;
      for (int j=0;j<size;j++)
	{
	  GridWorld gw(filename,i,j);
	  gw.generateVectorStates();
	  torch::Tensor output = net->actorOutput(torch::tensor(gw.getCurrentState().getStateVector()).unsqueeze(0));
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
  EpisodePlayerGW ep(FLAGS_map);
  ep.displayOnGrid(toDisplay);
  a.exec();
}

void Commands::generateDataSetGW()
{
  ToolsGW t;
  t.generateDataSet(FLAGS_mp,FLAGS_nmaps,FLAGS_n,FLAGS_T,FLAGS_wp);
}


void Commands::learnForwardModelGW()
{
  GridWorld gw;
  ToolsGW t(gw);
  string path = FLAGS_mp;
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

  int nTe = stateInputsTe.size(0), T = stateInputsTe.size(1), s = stateInputsTe.size(3);

  stateInputsTe = stateInputsTe.reshape({nTe*T,3,s,s});
  stateLabelsTe = stateLabelsTe.reshape({nTe*T,3,s,s});
  actionInputsTe = actionInputsTe.reshape({nTe*T,4});
  rewardLabelsTe = rewardLabelsTe.reshape({nTe*T});
  
  if (FLAGS_wn)
    {
      actionInputsTr+=torch::zeros({actionInputsTr.size(0),T,4}).normal_(0,FLAGS_sd);
    }
  actionInputsTr = torch::softmax(actionInputsTr,2);
  ForwardGW forwardModel(stateInputsTr.size(3),FLAGS_sc1);
  forwardModel->to(torch::Device(torch::kCUDA));
  ModelBased<GridWorld,ForwardGW, PlannerGW> agent(gw,forwardModel);
  torch::optim::Adam optimizer(forwardModel->parameters(),FLAGS_lr); 
  
  while(true)
    {
      agent.learnForwardModel(&optimizer, actionInputsTr, stateInputsTr,stateLabelsTr, rewardLabelsTr,FLAGS_n,FLAGS_bs, FLAGS_beta, FLAGS_asp);
      agent.saveTrainingData();
      torch::save(agent.getForwardModel(),FLAGS_mdl+".pt");
      agent.getForwardModel()->saveParams(FLAGS_mdl+"_Params");
      //Computing accuracy
      {
	torch::NoGradGuard no_grad;
	auto model = agent.getForwardModel();
	model->forward(stateInputsTe.to(model->getUsedDevice()),actionInputsTe.to(model->getUsedDevice()));
	t.rewardAccuracy(model->predictedReward.to(torch::Device(torch::kCPU)),rewardLabelsTe); 
	t.transitionAccuracy(model->predictedState.to(torch::Device(torch::kCPU)),stateLabelsTe);    
      }
    }
}

void Commands::playModelBasedGW(int argc, char* argv[])
{
  //QApplication a(argc,argv);
  ForwardGW fm(FLAGS_mdl+"_Params");
  torch::load(fm,FLAGS_mdl+".pt");
  GridWorld gw(FLAGS_map);
  ModelBased<GridWorld,ForwardGW,PlannerGW> agent(gw,fm,PlannerGW());
  agent.gradientBasedPlanner(FLAGS_K,FLAGS_T,FLAGS_gs,FLAGS_lr);
  //  agent.playOne(FLAGS_K,FLAGS_T,FLAGS_gs,FLAGS_lr);
  // EpisodePlayerGW ep(FLAGS_map);
  //ep.playEpisode(agent.getWorld().getStateSequence());
  //a.exec();
}


void Commands::tc1()
{
  /*
  ForwardGW fm("../temp/ForwardGW_Params");
  torch::load(fm,"../temp/ForwardGW.pt");
  GridWorld gw("../GridWorld/Maps/Easy8x8/test/map0",7,6);
  gw.generateVectorStates();
  ModelBased<GridWorld,ForwardGW,PlannerGW> agent(gw,fm,PlannerGW());
  agent.gradientBasedPlanner(FLAGS_K,FLAGS_T,FLAGS_gs,FLAGS_lr);
  */

  ForwardGW fm(FLAGS_mdl+"_Params");
  torch::load(fm,FLAGS_mdl+".pt");
  GridWorld gw("../GridWorld/Maps/Hard8x8/train/map0",4,2);
  gw.generateVectorStates();
  ModelBased<GridWorld,ForwardGW,PlannerGW> agent(gw,fm,PlannerGW());
  ofstream f("../temp/frgw");
    for (int i=0;i<1000;i++)
      {
	torch::Tensor a = torch::tensor({1-(i/1000.),i/1000.,0.,0.}).to(torch::kFloat32);
	torch::Tensor s = torch::tensor(gw.getCurrentState().getStateVector());
	fm->forward(s.unsqueeze(0),a.unsqueeze(0).to(fm->getUsedDevice()));
	f<<*fm->predictedReward.to(torch::Device(torch::kCPU)).data<float>()<<endl;    
      }
}

void Commands::tc2()
{
  ForwardGW fm("../temp/ForwardGW_Params");
  torch::load(fm,"../temp/ForwardGW.pt");
  ofstream f("../temp/gbpAcc");
  GridWorld gw(FLAGS_map);
  gw.generateVectorStates();  
  ModelBased<GridWorld,ForwardGW,PlannerGW> agent(gw,fm,PlannerGW());
  for (int i=0;i<FLAGS_n;i++)
    {      
      agent.playOne(FLAGS_K,FLAGS_T,FLAGS_gs,FLAGS_lr);
      f<<agent.rewardHistory().back()<<endl;
      agent.resetWorld();
    }
}

void Commands::tc3()
{

}

void Commands::generateMapSS()
{
  MapSS map(800);
  map.generate(FLAGS_nplan,FLAGS_pmin,FLAGS_pmax,FLAGS_nwp,FLAGS_rwp);
  map.save(FLAGS_map);
}

void Commands::generateMapPoolSS()
{
  MapSS map(800); 
  map.generateMapPool(FLAGS_nplan,FLAGS_pmin,FLAGS_pmax,FLAGS_nwp,FLAGS_rwp,FLAGS_mp,FLAGS_nmaps);
}


void Commands::showMapSS(int argc, char* argv[])
{
  QApplication a(argc,argv);
  EpisodePlayerSS ep(FLAGS_map);
  ep.showMap();
  a.exec();
}

void Commands::playRandomSS(int argc, char* argv[])
{
  MapSS map(800);
  map.generate(FLAGS_nplan,FLAGS_pmin,FLAGS_pmax,FLAGS_nwp,FLAGS_rwp);
  map.save("../Starship/Maps/test");
  SpaceWorld sw("../Starship/Maps/test");
  for (int i=0;i<FLAGS_n;i++)
    {
      vector<float> a = sw.randomAction();
      a[2] = i/20.;
      sw.setTakenAction(a);
      sw.transition();
    }
  QApplication a(argc,argv);
  EpisodePlayerSS ep("../Starship/Maps/test");
  ep.playEpisode(sw.getActionSequence(), sw.getStateSequence(), SHIP_MAX_THRUST);  
  a.exec();
}

void Commands::generateDataSetSS()
{
  ToolsSS t;
  t.generateDataSet(FLAGS_mp,FLAGS_nmaps,FLAGS_n,FLAGS_T,FLAGS_trp, FLAGS_wp, FLAGS_esp);
}

void Commands::learnForwardModelSS()
{
  SpaceWorld sw;
  string path = FLAGS_mp;
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

  int nTr = stateInputsTr.size(0), nTe = stateInputsTe.size(0), T = stateInputsTe.size(1), s = stateInputsTe.size(2);
  stateInputsTe = stateInputsTe.reshape({nTe*T,s});
  stateLabelsTe = stateLabelsTe.reshape({nTe*T,4});
  actionInputsTe = actionInputsTe.reshape({nTe*T,6});
  rewardLabelsTe = rewardLabelsTe.reshape({nTe*T});
  
  if (FLAGS_wn)
    {
      actionInputsTr+=torch::cat({torch::zeros({actionInputsTr.size(0),T,4}).normal_(0,FLAGS_sd),torch::zeros({actionInputsTr.size(0),T,2})},2);
    }
  ForwardSS forwardModel(stateInputsTr.size(2),512,2);
  //ForwardSS forwardModel(FLAGS_mdl+"_Params");
  //torch::load(forwardModel,FLAGS_mdl+".pt");  
  forwardModel->to(torch::Device(torch::kCUDA));
  ModelBased<SpaceWorld,ForwardSS, PlannerGW> agent(sw,forwardModel);
  torch::optim::Adam optimizer(forwardModel->parameters(),FLAGS_lr); 
  int l=0;
  ofstream ftrp("../temp/trp_mse"+FLAGS_tag);
  ofstream ftrv("../temp/trv_mse"+FLAGS_tag);  
  ofstream ftep("../temp/tep_mse"+FLAGS_tag);
  ofstream ftev("../temp/tev_mse"+FLAGS_tag);  
  ofstream ftrr("../temp/trr_mse"+FLAGS_tag);
  ofstream fter("../temp/ter_mse"+FLAGS_tag);  
  
  while(l!=5000)
    {
      l++;
      agent.learnForwardModel(&optimizer, actionInputsTr, stateInputsTr,stateLabelsTr, rewardLabelsTr,FLAGS_n,FLAGS_bs, FLAGS_beta, FLAGS_asp);
      if (l%500 == 0)
	{
	  torch::save(agent.getForwardModel(),FLAGS_mdl+"cp"+to_string(l)+".pt");
	  agent.getForwardModel()->saveParams(FLAGS_mdl+"cp"+to_string(l)+"_Params");
	  cout<<"Checkpointing..."<<endl;
	}
      agent.saveTrainingData();
      torch::save(agent.getForwardModel(),FLAGS_mdl+".pt");
      agent.getForwardModel()->saveParams(FLAGS_mdl+"_Params");
      //Computing accuracy
      {	
	torch::NoGradGuard no_grad;
	ToolsSS t;
	auto model = agent.getForwardModel();
	int splitSize = 10000;
	vector<torch::Tensor> sitrSplit = torch::split(stateInputsTr.reshape({nTr*T,s}),splitSize,0);
	vector<torch::Tensor> aitrSplit = torch::split(actionInputsTr.reshape({nTr*T,6}),splitSize,0);
	vector<torch::Tensor> sltrSplit = torch::split(stateLabelsTr.reshape({nTr*T,4}),splitSize,0);
	vector<torch::Tensor> rltrSplit = torch::split(rewardLabelsTr.reshape({nTr*T}),splitSize,0);
	unsigned int nSplit = sitrSplit.size();
	
	for (unsigned int i=0;i<nSplit;i++)
	  {
	    model->forward(sitrSplit[i].to(model->getUsedDevice()),aitrSplit[i].to(model->getUsedDevice()));
	    t.transitionAccuracy(t.normalize(model->predictedState,true),sltrSplit[i],nSplit);
	    t.rewardAccuracy(model->predictedReward.to(torch::Device(torch::kCPU)),rltrSplit[i], nSplit);
	  }
	
	ftrp<<pow(*t.pMSE.data<float>(),0.5)<<endl;
	ftrv<<pow(*t.vMSE.data<float>(),0.5)<<endl;
	ftrr<<pow(*t.rMSE.data<float>(),0.5)<<endl;	
	t.displayTAccuracy(nTr*T);
	t.displayRAccuracy();

        t = ToolsSS();
	vector<torch::Tensor> siteSplit = torch::split(stateInputsTe,splitSize,0);
	vector<torch::Tensor> aiteSplit = torch::split(actionInputsTe,splitSize,0);
	vector<torch::Tensor> slteSplit = torch::split(stateLabelsTe,splitSize,0);
	vector<torch::Tensor> rlteSplit = torch::split(rewardLabelsTe,splitSize,0);	
        nSplit = siteSplit.size();
	
	for (unsigned int i=0;i<nSplit;i++)
	  {
	    model->forward(siteSplit[i].to(model->getUsedDevice()),aiteSplit[i].to(model->getUsedDevice()));
	    t.transitionAccuracy(t.normalize(model->predictedState,true),slteSplit[i],nSplit);
	    t.rewardAccuracy(model->predictedReward.to(torch::Device(torch::kCPU)),rlteSplit[i], nSplit);	
	  }
	
	ftep<<pow(*t.pMSE.data<float>(),0.5)<<endl;
	ftev<<pow(*t.vMSE.data<float>(),0.5)<<endl;
	fter<<pow(*t.rMSE.data<float>(),0.5)<<endl;
	t.displayTAccuracy(nTe*T);
	t.displayRAccuracy();
	
	//	model->forward(stateInputsTe.to(model->getUsedDevice()),actionInputsTe.to(model->getUsedDevice()));

	//	fte << torch::mse_loss(ToolsSS().normalize(model->predictedState,true), stateLabelsTe) << endl;
	
	//ToolsSS().rewardAccuracy(model->predictedReward.to(torch::Device(torch::kCPU)),rewardLabelsTe);	
	//ToolsSS().transitionAccuracy(ToolsSS().normalize(model->predictedState,true).to(torch::Device(torch::kCPU)),stateLabelsTe);
      }

    }  
}

void Commands::playModelBasedSS(int argc, char* argv[])
{
  ForwardSS fm(FLAGS_mdl+"_Params");
  torch::load(fm,FLAGS_mdl+".pt");
  SpaceWorld sw(FLAGS_map);
  sw.repositionShip(sw.getWaypoints()[2].getCentre());
  ModelBased<SpaceWorld,ForwardSS,PlannerGW> agent(sw,fm);
  agent.playOne(FLAGS_K,FLAGS_T,FLAGS_gs,FLAGS_lr);
  QApplication a(argc,argv);
  EpisodePlayerSS ep(FLAGS_map);
  ep.playEpisode(agent.getWorld().getActionSequence(),agent.getWorld().getStateSequence(), SHIP_MAX_THRUST);
  a.exec();
}
