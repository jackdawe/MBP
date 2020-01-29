#include "commands.h"

DEFINE_string(mdl,"../temp/model","Path to a model file. Do not add the .pt extension.");
DEFINE_string(tag,"","Suffix for auto generated files"); 
DEFINE_string(f,"../file","The path to a file");
DEFINE_string(dir,"../temp/","The path to a directory (must end with a /)");
DEFINE_string(map,"","Path to a map file");
DEFINE_string(mp,"","Path to a directory containing your maps");
DEFINE_int32(nmaps,1,"The number of maps in the map pool directory");
DEFINE_string(seed,"","Path to the .pt file containing the seed tensor"); 

//GridWorld flags

DEFINE_int32(size,8,"The generated maps will be of size sizexsize");
DEFINE_int32(maxObst,1,"The maps will be generated having a random number of obstacles between 1 and maxObst");

//Starship flags

DEFINE_int32(nplan,N_PLANETS,"Number of planets for mapss generation");
DEFINE_int32(pmin,PLANET_MIN_SIZE,"Planet minimum radius for mapss generation");
DEFINE_int32(pmax,PLANET_MAX_SIZE,"Planet maximum radius for mapss generation");
DEFINE_int32(nwp,N_WAYPOINTS,"Number of waypoints for mapss generation");
DEFINE_int32(rwp,WAYPOINT_RADIUS,"Waypoint radius for mapss generation");
DEFINE_double(trp,0.9,"Share of the training set from the whole dataset");
DEFINE_double(px,-1,"ship x coordinate");
DEFINE_double(py,-1,"ship y coordinate");

//Model flags

DEFINE_int32(sc1,16,"Number of feature maps of the first conv layer of the encoder. Next layers have twice as many features maps and the NN is shaped accordingly");

//Learning parameters flags

DEFINE_double(eps,0.1,"Probability of exploring for agents using epsilon greedy policies");
DEFINE_double(g,0.95,"Discount factor");
DEFINE_double(lr,0.001,"Learning Rate");
DEFINE_double(beta,1,"Coefficient applied to the entropy loss");
DEFINE_double(zeta,1,"Coefficient applied to the value loss");
DEFINE_int32(bs,32,"Batch Size");

//Planning flags

DEFINE_int32(K,1,"Number of rollouts");
DEFINE_int32(T,1,"Number of timesteps to unroll");
DEFINE_int32(gs,1,"Number of gradient steps");


DEFINE_bool(asp,true,"If true, all input states are provided for training for model based agent. If false, only initial state and the action sequence are provided and the agent uses his predicted states to predict the next state"); 
DEFINE_int32(n,10000,"Number of training episodes");
DEFINE_double(wp,0.1,"Percentage of forced win scenarios during the dataset generation"); 
DEFINE_bool(wn,false,"Adding white noise to the one-hot encoded action vectors");
DEFINE_double(sd,0.25,"Standard deviation");

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
  t.generateDataSet(FLAGS_mp,FLAGS_nmaps,FLAGS_n,FLAGS_T, FLAGS_trp, FLAGS_wp);
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
  //  actionInputsTr = torch::softmax(actionInputsTr,2);
  ForwardGW forwardModel(stateInputsTr.size(3),FLAGS_sc1);
  forwardModel->to(torch::Device(torch::kCUDA));
  ModelBased<GridWorld,ForwardGW, PlannerGW> agent(gw,forwardModel);
  torch::optim::Adam optimizer(forwardModel->parameters(),FLAGS_lr); 
  int l=0;
  while(l!=50)
    {
      l++;
      agent.learnForwardModel(&optimizer, actionInputsTr, stateInputsTr,stateLabelsTr, rewardLabelsTr,FLAGS_n,FLAGS_bs, FLAGS_beta, FLAGS_asp);
      agent.saveTrainingData();
      torch::save(agent.getForwardModel(),FLAGS_mdl+".pt");
      agent.getForwardModel()->saveParams(FLAGS_mdl+"_Params");
      //Computing accuracy
      {
	torch::NoGradGuard no_grad;
	auto model = agent.getForwardModel();
	model->forward(stateInputsTe.to(model->usedDevice),actionInputsTe.to(model->usedDevice));
	t.rewardAccuracy(model->predictedReward.to(torch::Device(torch::kCPU)),rewardLabelsTe); 
	t.transitionAccuracy(model->predictedState.to(torch::Device(torch::kCPU)),stateLabelsTe);    
      }
    }
}

void Commands::playModelBasedGW(int argc, char* argv[])
{
  ForwardGW fm(FLAGS_mdl+"_Params");
  torch::load(fm,FLAGS_mdl+".pt");
  GridWorld gw(FLAGS_map);
  ModelBased<GridWorld,ForwardGW,PlannerGW> agent(gw,fm,PlannerGW());
  //  agent.gradientBasedPlanner(FLAGS_K,FLAGS_T,FLAGS_gs,FLAGS_lr);
  //  agent.playOne(FLAGS_K,FLAGS_T,FLAGS_gs,FLAGS_lr);
  QApplication a(argc,argv);
  EpisodePlayerGW ep(FLAGS_map);
  ep.playEpisode(agent.getWorld().getStateSequence());
  a.exec();
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
  ModelBased<GridWorld,ForwardGW,PlannerGW> agent(gw,fm,PlannerGW());
  ofstream f("../temp/frgw");
    for (int i=0;i<1000;i++)
      {
	torch::Tensor a = torch::tensor({1-(i/1000.),i/1000.,0.,0.}).to(torch::kFloat32);
	torch::Tensor s = torch::tensor(gw.getCurrentState().getStateVector());
	fm->forward(s.unsqueeze(0),a.unsqueeze(0).to(fm->usedDevice));
	f<<*fm->predictedReward.to(torch::Device(torch::kCPU)).data<float>()<<endl;    
      }
}

void Commands::tc2()
{
  ForwardGW fm("../temp/ForwardGW_Params");
  torch::load(fm,"../temp/ForwardGW.pt");
  ofstream f("../temp/gbpAcc");
  GridWorld gw(FLAGS_map);
  ModelBased<GridWorld,ForwardGW,PlannerGW> agent(gw,fm,PlannerGW());
  for (int i=0;i<FLAGS_n;i++)
    {      
      //      agent.playOne(FLAGS_K,FLAGS_T,FLAGS_gs,FLAGS_lr);
      f<<agent.rewardHistory().back()<<endl;
      agent.resetWorld();
    }
}

void Commands::tc3()
{
  ForwardGW forwardModel(8,FLAGS_sc1);
  cout<<forwardModel<<endl;
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
  t.generateDataSet(FLAGS_mp,FLAGS_nmaps,FLAGS_n,FLAGS_T,FLAGS_trp, FLAGS_wp);
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
  if (FLAGS_wn)
    {
      actionInputsTr+=torch::cat({torch::zeros({actionInputsTr.size(0),T,4}).normal_(0,FLAGS_sd),torch::zeros({actionInputsTr.size(0),T,2})},2);
    }
  ForwardSS forwardModel(stateInputsTr.size(2),512,2);
  forwardModel->to(torch::Device(torch::kCUDA)); //OOIEJFOEWJFOI
  //ForwardSS forwardModel(FLAGS_mdl+"_Params");
  //  torch::load(forwardModel,FLAGS_mdl+".pt");  
  ModelBased<SpaceWorld,ForwardSS, PlannerGW> agent(sw,forwardModel);
  torch::optim::Adam optimizer(forwardModel->parameters(),FLAGS_lr); 
  int l=0;
  ofstream ftrp("../temp/trp_mse"+FLAGS_tag);
  ofstream ftrv("../temp/trv_mse"+FLAGS_tag);  
  ofstream ftep("../temp/tep_mse"+FLAGS_tag);
  ofstream ftev("../temp/tev_mse"+FLAGS_tag);  
  ofstream ftrr("../temp/trr_mse"+FLAGS_tag);
  ofstream fter("../temp/ter_mse"+FLAGS_tag);  
  
  while(l!=40000)
    {
      l++;
      agent.learnForwardModel(&optimizer, actionInputsTr, stateInputsTr,stateLabelsTr, rewardLabelsTr,FLAGS_n,FLAGS_bs, FLAGS_beta, FLAGS_asp);
      if (l%40 == 0)
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
	int splitSize = 1000;
	/*	vector<torch::Tensor> sitrSplit = torch::split(stateInputsTr,splitSize,0);
	vector<torch::Tensor> aitrSplit = torch::split(actionInputsTr,splitSize,0);
	vector<torch::Tensor> sltrSplit = torch::split(stateLabelsTr,splitSize,0);
	vector<torch::Tensor> rltrSplit = torch::split(rewardLabelsTr,splitSize,0);
	unsigned int nSplit = sitrSplit.size();
	for (unsigned int i=0;i<nSplit;i++)
	  {
	    int nSpl = sitrSplit[i].size(0); 
	    if (FLAGS_asp)
	      {		
		model->forward(sitrSplit[i].reshape({nSpl*T,s}),aitrSplit[i].reshape({nSpl*T,6}));
	      }
	    else
	      {
		torch::Tensor predictedStates = torch::zeros({T,nSpl,s}).to(model->usedDevice);
		torch::Tensor predictedRewards = torch::zeros({T,nSpl}).to(model->usedDevice);
		model->forward(sitrSplit[i].transpose(0,1)[0],aitrSplit[i].transpose(0,1)[0]);
		predictedStates[0] = forwardModel->predictedState;      
	        predictedRewards[0] = forwardModel->predictedReward;
		for (int t=1;t<T;t++)
		  {
		    model->forward(predictedStates[t-1],aitrSplit[i].transpose(0,1)[t]);
		    predictedStates[t] = forwardModel->predictedState;      
		    predictedRewards[t] = forwardModel->predictedReward;		    
		  }
	        model->predictedState = predictedStates.transpose(0,1).reshape({nSpl*T,s}); 
	        model->predictedReward = predictedRewards.transpose(0,1).reshape({nSpl*T});	       
	      }
	    t.transitionAccuracy(model->predictedState,sltrSplit[i].reshape({nSpl*T,4}).to(model->usedDevice),nSplit,false);
	    t.rewardAccuracy(model->predictedReward,rltrSplit[i].reshape({nSpl*T}).to(model->usedDevice), nSplit,false);
	  }
	ftrp<<pow(*t.pMSE.data<float>(),0.5)<<endl;
	ftrv<<pow(*t.vMSE.data<float>(),0.5)<<endl;
	ftrr<<pow(*t.rMSE.data<float>(),0.5)<<endl;	
	t.displayTAccuracy(nTr*T);
	t.displayRAccuracy();
	*/
	//        t = ToolsSS(); 
	vector<torch::Tensor> siteSplit = torch::split(stateInputsTe,splitSize,0);
	vector<torch::Tensor> aiteSplit = torch::split(actionInputsTe,splitSize,0);
	vector<torch::Tensor> slteSplit = torch::split(stateLabelsTe,splitSize,0);
	vector<torch::Tensor> rlteSplit = torch::split(rewardLabelsTe,splitSize,0);	
        unsigned nSplit = siteSplit.size();
	
	for (unsigned int i=0;i<nSplit;i++)
	  {
	    int nSpl = siteSplit[i].size(0); 
	    if (FLAGS_asp)
	      {
		model->forward(siteSplit[i].reshape({nSpl*T,s}),aiteSplit[i].reshape({aiteSplit[i].size(0)*T,6}));
	      }
	    else
	      {
		torch::Tensor predictedStates = torch::zeros({T,nSpl,s}).to(model->usedDevice);
		torch::Tensor predictedRewards = torch::zeros({T,nSpl}).to(model->usedDevice);
	        model->forward(siteSplit[i].transpose(0,1)[0],aiteSplit[i].transpose(0,1)[0]);
		predictedStates[0] = forwardModel->predictedState;      
	        predictedRewards[0] = forwardModel->predictedReward;
		for (int t=1;t<T;t++)
		  {
		    model->forward(predictedStates[t-1],aiteSplit[i].transpose(0,1)[t]);
		    predictedStates[t] = forwardModel->predictedState;      
		    predictedRewards[t] = forwardModel->predictedReward;		    
		  }
	        model->predictedState = predictedStates.transpose(0,1).reshape({nSpl*T,s}); 
	        model->predictedReward = predictedRewards.transpose(0,1).reshape({nSpl*T});	       
	      }
	    t.transitionAccuracy(model->predictedState,slteSplit[i].reshape({nSpl*T,4}).to(model->usedDevice),nSplit,true);
	    t.rewardAccuracy(model->predictedReward,rlteSplit[i].reshape({nSpl*T}).to(model->usedDevice), nSplit,true);
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

void Commands::generateSeedSS()
{
  ToolsSS().generateSeed(FLAGS_T,FLAGS_K,FLAGS_seed);
}

void Commands::playModelBasedSS(int argc, char* argv[])
{
  ForwardSS fm(FLAGS_mdl+"_Params");
  torch::load(fm,FLAGS_mdl+".pt");
  SpaceWorld sw(FLAGS_map);
  
  if (FLAGS_px != -1 && FLAGS_py != -1)
    {
      sw.repositionShip(Vect2d(FLAGS_px,FLAGS_py));
    }
  ModelBased<SpaceWorld,ForwardSS,PlannerGW> agent(sw,fm);
  torch::Tensor actions = torch::zeros(0);
  if (FLAGS_seed != "")
    {
      torch::load(actions,FLAGS_seed);
    }
  agent.playOne(torch::tensor(agent.currentState().getStateVector()),sw.getActions(),FLAGS_K,FLAGS_T,FLAGS_gs,FLAGS_lr,actions);
  QApplication a(argc,argv);
  EpisodePlayerSS ep(FLAGS_map);
  ep.playEpisode(agent.getWorld().getActionSequence(),agent.getWorld().getStateSequence(), SHIP_MAX_THRUST);
  a.exec();
}

void Commands::testModelBasedSS()
{
  /*  ForwardSS fm(FLAGS_mdl+"_Params");
  torch::load(fm,FLAGS_mdl+".pt");
  ofstream f("../temp/gbpAccSS");
  SpaceWorld sw(FLAGS_mp, FLAGS_nmaps);
  ModelBased<SpaceWorld,ForwardSS,PlannerGW> agent(sw,fm,PlannerGW());
  for (int i=0;i<FLAGS_n;i++)
    {      
      agent.playOne(FLAGS_K,FLAGS_T,FLAGS_gs,FLAGS_lr);
      f<<agent.rewardHistory().back()<<endl;
      agent.resetWorld();
    }*/
  int T=40;
  string path=FLAGS_mp;
  torch::Tensor stateInputsTe, actionInputsTe, stateLabelsTe, rewardLabelsTe;
  torch::load(stateInputsTe,path+"stateInputsTest.pt");
  torch::load(actionInputsTe, path+"actionInputsTest.pt");
  torch::load(stateLabelsTe,path+"stateLabelsTest.pt");
  torch::load(rewardLabelsTe, path+"rewardLabelsTest.pt");
  cout<<stateInputsTe.slice(0,1,1,1)<<endl;
  torch::Tensor bsi = stateInputsTe[FLAGS_n].unsqueeze(0);
  torch::Tensor bai = actionInputsTe[FLAGS_n].unsqueeze(0);
  torch::Tensor bsl = stateLabelsTe[FLAGS_n].unsqueeze(0);
  torch::Tensor brl = rewardLabelsTe[FLAGS_n].unsqueeze(0);
  ForwardSS forwardModel(FLAGS_mdl+"_Params");
  torch::load(forwardModel,FLAGS_mdl+".pt");
  torch::Tensor predictedStates = torch::zeros({T,1,16});
  torch::Tensor predictedRewards = torch::zeros({T,1});
  forwardModel->forward(bsi.transpose(0,1)[0],bai.transpose(0,1)[0]);
  predictedStates[0] = forwardModel->predictedState;      
  predictedRewards[0] = forwardModel->predictedReward;
  ofstream f("../temp/file");
  cout<<stateInputsTe[0]<<endl;
  cout<<stateLabelsTe[0]<<endl;
  cout<<actionInputsTe[0]<<endl;  
  cout<<rewardLabelsTe[0]<<endl;
  for (int o=0;o<actionInputsTe.flatten().size(0);o++)
    {
      //      f<<*actionInputsTe.slice(-1,4,6,1).flatten()[o].data<float>()<<endl;
    }
  for (int t=1;t<T;t++)
    {
      forwardModel->forward(predictedStates[t-1],bai.transpose(0,1)[t]);
      predictedStates[t] = forwardModel->predictedState;      
      predictedRewards[t] = forwardModel->predictedReward;		    
    }
  forwardModel->predictedState = predictedStates.transpose(0,1);
  forwardModel->predictedReward = predictedRewards.transpose(0,1);

  cout<<torch::cat({bsl,predictedStates.transpose(0,1).slice(2,0,4,1)},2)<<endl;
  cout<<torch::cat({brl.unsqueeze(2),predictedRewards.transpose(0,1).unsqueeze(2)},2)<<endl;  
  cout<<ToolsSS().moduloMSE(bsl.slice(2,0,2,1),predictedStates.transpose(0,1).slice(2,0,2,1),false).pow(0.5)<<endl;
}

void Commands::tc4()
{
  string path=FLAGS_mp;
  torch::Tensor stateInputsTe, actionInputsTe, stateLabelsTe, rewardLabelsTe;
  torch::load(stateInputsTe,path+"stateInputsTest.pt");
  torch::load(actionInputsTe, path+"actionInputsTest.pt");
  torch::load(stateLabelsTe,path+"stateLabelsTest.pt");
  torch::load(rewardLabelsTe, path+"rewardLabelsTest.pt");
  torch::Tensor bsi = stateInputsTe.slice(0,0,2000,1);
  torch::Tensor bai = actionInputsTe.slice(0,0,2000,1);
  torch::Tensor bsl = stateLabelsTe.slice(0,0,2000,1);
  torch::Tensor brl = rewardLabelsTe.slice(0,0,2000,1);
  ForwardSS forwardModel(FLAGS_mdl+"_Params");
  torch::load(forwardModel,FLAGS_mdl+".pt");
  int n=bsi.size(0), T=bsi.size(1);
  torch::Tensor predictedStates = torch::zeros({T,n,16});
  torch::Tensor predictedRewards = torch::zeros({T,n});
  forwardModel->forward(bsi.transpose(0,1)[0],bai.transpose(0,1)[0]);
  predictedStates[0] = forwardModel->predictedState;
  predictedRewards[0] = forwardModel->predictedReward;
  ofstream f("../temp/file");
  for (int o=0;o<actionInputsTe.flatten().size(0);o++)
    {
      //      f<<*actionInputsTe.slice(-1,4,6,1).flatten()[o].data<float>()<<endl;
    }
  for (int t=1;t<T;t++)
    {
      forwardModel->forward(predictedStates[t-1],bai.transpose(0,1)[t]);
      predictedStates[t] = forwardModel->predictedState;      
      predictedRewards[t] = forwardModel->predictedReward;		    
    }
  torch::Tensor x = predictedStates.transpose(0,1).slice(-1,0,2,1);
  torch::Tensor y = bsl.slice(-1,0,2,1);  
  forwardModel->predictedReward = predictedRewards.transpose(0,1);

  torch::Tensor x1 = x;
  torch::Tensor y1 = y;
  torch::Tensor compare = torch::cat({x1.unsqueeze(0),y1.unsqueeze(0)},0);
  torch::Tensor mini = get<0>(torch::min(compare,0));
  torch::Tensor maxi = get<0>(torch::max(compare,0));
  torch::Tensor a = (mini+800-maxi).unsqueeze(0);
  compare = torch::cat({a,800-a});
  torch::Tensor mse = get<0>(torch::min(compare,0)).mean(1).flatten();
  for (int i=0;i<mse.size(0);i++)
    {
      f<<*mse[i].to(torch::Device(torch::kCPU)).data<float>()<<endl;
      if (*mse[i].data<float>()>390)
	{
	  //	  cout<<torch::cat({x.slice(-1,0,2,1).flatten()[i].unsqueeze(0),bsl.slice(-1,0,2,1).flatten()[i].unsqueeze(0)})<<endl;
	}
    }
}

