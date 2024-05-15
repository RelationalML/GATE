from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

#from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    #torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
    is_sparse,
)

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures,LargestConnectedComponents
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import segregate_self_loops,dense_to_sparse,\
index_to_mask,get_laplacian,erdos_renyi_graph,to_networkx
from torchmetrics.functional import pairwise_cosine_similarity
from torchmetrics import AUROC
import torch
from torch.nn import Linear
import torch.nn.functional as F 
#from torch_geometric.nn import GCNConv,GATv2Conv
import numpy as np
import matplotlib.pyplot as plt
import copy
import json
import pprint
import pickle
from scipy import stats
import pandas as pd
import os.path
from math import floor,ceil
from itertools import product
import scipy.sparse as sp
from torch.distributions.multivariate_normal import MultivariateNormal
from tsne_torch import TorchTSNE as TSNE
from sklearn.cluster import KMeans
import heapq
from scipy import stats
from msgpass import MessagePassing
from torchmetrics.functional.pairwise.helpers import _check_input


path = 'ExpResults_MLP/'
device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def getDataHet(datasetName,splitID=1):

    print("Loading datasets as npz-file..")
    data = np.load('data/heterophilous-graphs/'+datasetName+'.npz')
    x = torch.tensor(data['node_features'], dtype=torch.float)
    y = torch.tensor(data['node_labels'], dtype=torch.long)
    edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
    train_mask = torch.tensor(data['train_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    val_mask = torch.tensor(data['val_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    test_mask = torch.tensor(data['test_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    num_classes = len(torch.unique(y))
    num_targets = 1 if num_classes == 2 else num_classes
    
    print("Converting to PyG dataset...")
    data = Data(x=x, edge_index=edge_index)
    data.y = y
    data.num_classes = num_classes
    data.num_targets = num_targets
    data.train_mask = train_mask[:,splitID] #split_idx = 1, 10 splits provided in dataset
    data.val_mask = val_mask[:,splitID]
    data.test_mask = test_mask[:,splitID]
    return data,data.num_features,data.num_classes

def getData(datasetName, dataTransform, randomLabels=False,oneHotFeatures=False,randomLabelCount=None,splitID=1):
    
    if datasetName[:3]=='Syn': # == 'Synthetic':
        synID = datasetName.split("_")[1]
        with open('SyntheticData/D'+str(synID)+'.pkl', 'rb') as f:
            data = pickle.load(f)
            return data,data.x.shape[1],len(torch.unique(data.y))
    if datasetName in ['Cora','Citeseer','Pubmed']:
        dataset = Planetoid(root='data/Planetoid', name=datasetName, transform=NormalizeFeatures())
        data = dataset[0]
        if dataTransform=='removeIsolatedNodes':
            out = segregate_self_loops(data.edge_index)
            edge_index, edge_attr, loop_edge_index, loop_edge_attr = out
            mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
            mask[edge_index.view(-1)] = 1
            data.train_mask = data.train_mask & mask
            data.val_mask = data.val_mask & mask
            data.test_mask = data.test_mask & mask
        if dataTransform=='useLCC':
            transformLCC = LargestConnectedComponents()
            data = transformLCC(data)
        if randomLabels:
            if randomLabelCount==None:
                data.y = torch.randint(low=0,high=len(torch.unique(data.y)),size=data.y.shape)
            else:
                data.y = torch.randint(low=0,high=randomLabelCount,size=data.y.shape)
            
        if oneHotFeatures:
            data.x = torch.tensor(F.one_hot(data.y).clone().detach(),dtype = torch.float32)
        
        return data,data.x.shape[1],len(torch.unique(data.y))#dataset.num_features,dataset.num_classes
    else:
        s = datasetName.split("_")
        if len(s)==1:
            s = s+[str(splitID)]
        return getDataHet(s[0],int(s[1]))


class MLP(torch.nn.Module):
    def __init__(self, numLayers, dims, bias=False, activation='relu',init='LLortho'):
        super().__init__()
        self.numLayers = numLayers
        self.bias=bias
        if activation=='relu':
            self.activation = F.relu
        self.layers = torch.nn.ModuleList(
            [Linear(dims[j],dims[j+1],bias=self.bias) 
                       for j in range(self.numLayers)])
        self.initialize(init)
    
    def forward(self, x):
        for l in range(numLayers):
            x = self.layers[l](x)
            if l<(numLayers-1):
                x = self.activation(x)
        return x
    
    def initialize(self,init='LLortho'):
        
        if init=='LLortho':
            dim = self.layers[0].weight.shape
            submatrix = torch.nn.init.orthogonal_(torch.empty(ceil(dim[0]/2),ceil(dim[1]),device=device))
            self.layers[0].weight.data = torch.cat((submatrix, -submatrix), dim=0)
            
            dim = self.layers[self.numLayers-1].weight.shape
            submatrix = torch.nn.init.orthogonal_(torch.empty(ceil(dim[0]),ceil(dim[1]/2),device=device))
            self.layers[self.numLayers-1].weight.data = torch.cat((submatrix, -submatrix), dim=1)

            for l in range(1,numLayers-1):
                dim = self.layers[l].weight.shape
                submatrixW = torch.nn.init.orthogonal_(torch.empty(ceil(dim[0]/2),ceil(dim[1]/2),device=device))
                submatrixW = torch.cat((submatrixW, -submatrixW), dim=0)
                submatrixW = torch.cat((submatrixW, -submatrixW), dim=1)
                self.layers[l].weight.data = submatrixW

            for l in range(numLayers):
                self.layers[l].weight.data.requires_grad = True

def computeStatSumry(arr,quantiles):
  r = {'mean': arr.mean(),
        'std': arr.std()}
  quantiles=torch.cat((torch.tensor([0,1],device=device),quantiles),dim=0)
  p = torch.quantile(arr,quantiles)
  r['min'] = p[0]
  r['max'] = p[1]
  for i in range(2,len(quantiles)):
    r[str(int(quantiles[i]*100))+'%ile'] = p[i]
  return r

def computeAlphaStatSumry(alphas,quantiles):
    return [computeStatSumry(alphas[1][np.where(np.equal(alphas[0][0],alphas[0][1])==True)[0]],quantiles),
       computeStatSumry(alphas[1][np.where(np.equal(alphas[0][0],alphas[0][1])==False)[0]],quantiles)]

def printExpSettings(expID,expSetting):
    print('Exp: '+str(expID))
    for k,v in expSetting.items():
        for k2,v2 in expSetting[k].items():
            if(k2==expID):
                print(k,': ',v2)

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


expSetting = pd.read_csv('ExpSettings_MLP.csv',index_col='expId').fillna('').to_dict()

#Add ExpIDs to run here corresponding to ExpSettings.csv 
expIDs = range(1,4+1)#[1055,1058] #Add ExpIDs to run here
runIDs = [] #If not specified, i.e. = [], then range(numRuns from settings file) will be set
trainLossToConverge =  0.00001
printLossEveryXEpoch = 1000
saveParamGradStatSumry = False
saveNeuronLevelL2Norms = False
saveLayerWiseForbNorms = False
saveWeightsAtMaxValAcc = False
saveNodeSmoothnessVals = False
saveNodeSmoothnessValsClassWise = False
saveAttentionParams = False
saveAlphas = False
saveAlphasSummary = False
saveBiases = False
saveOmega = False
getAlphaEntropy = False
randomLabels=False
oneHotFeatures = False

quantiles = torch.tensor((np.array(range(1,10,1))/10),dtype=torch.float32,device=device)
qLabels = [str(int(q*100))+'%ile' for q in quantiles]
labels = ['min','max','mean','std']+qLabels 

for expID in expIDs:
    numRuns = int(expSetting['numRuns'][expID])
    if len(runIDs)==0:
        runIDs = range(numRuns) #or specify
    datasetName = str(expSetting['dataset'][expID])
    optim = str(expSetting['optimizer'][expID])
    numLayers = int(expSetting['numLayers'][expID])
    numEpochs = int(expSetting['maxEpochs'][expID])
    lr = float(expSetting['initialLR'][expID])
    hiddenDims = [int(expSetting['hiddenDim'][expID])] * (numLayers-1)
    
    wghtDecay =  float(expSetting['wghtDecay'][expID])
    activation = str(expSetting['activation'][expID])
    dataTransform = str(expSetting['dataTransform'][expID]) #removeIsolatedNodes,useLCC 
    initScheme=str(expSetting['initScheme'][expID])
    scalScheme=str(expSetting['scalScheme'][expID])
    lrDecayFactor = float(expSetting['lrDecayFactor'][expID])
    useIdMap = bool(expSetting['useIdMap'][expID])
    useResLin = bool(expSetting['useResLin'][expID])
    if lrDecayFactor<1:
        lrDecayPatience = float(expSetting['lrDecayPatience'][expID])
    scalHPstr = [0,0,0]
    if len(str(expSetting['scalHP'][expID]))>0:
         scalHPstr=[float(x) for x in str(expSetting['scalHP'][expID]).split('|')] #e.g. (low,high) for uniform, (mean,std) for normal, (const) for const. Third parameter is beta
    
    bias = bool(expSetting['bias'][expID])

    if str(expSetting['Note'][expID])=='randLabels':
        randomLabels=True
    if str(expSetting['Note'][expID])=='OneHotFeats':
        oneHotFeatures = True
    if str(expSetting['Note'][expID])=='randLabels+OneHotFeats':
        randomLabels=True
        oneHotFeatures = True

    selfLoops = True

    
    print('*******')
    printExpSettings(expID,expSetting)
    print('*******')
    
    for run in runIDs:#range(numRuns):
        print('-- RUN ID: '+str(run))
        set_seeds(run)

        data,input_dim,output_dim = getData(datasetName,dataTransform,randomLabels,oneHotFeatures,run) 
        if output_dim==2:
            auroc = AUROC(task='binary')
        else:
            auroc = AUROC(task='multiclass',num_classes=output_dim)
        #output_dim +=1 # in case layer=1 for LL ortho sub
        data = data.to(device)
        
        dims = [input_dim]+hiddenDims+[output_dim]
        

        model = MLP(numLayers,dims,bias=bias,
                      activation=activation).to(device)
        # print(model)
        # for name,param in model.named_paramete,rs():
        #     print(name,param.shape)
        
        if optim=='SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wghtDecay)
        if optim=='Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wghtDecay)
        criterion = torch.nn.CrossEntropyLoss()
        if lrDecayFactor<1:
            lrScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=lrDecayFactor, patience=lrDecayPatience) #based on valAcc
        
        # define variables to track required elements
        trainLoss = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        valLoss = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        trainAcc = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        valAcc = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        testAcc = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        trainAUROC = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        valAUROC = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        testAUROC = torch.zeros(numEpochs, dtype=torch.float32, device = device)
            
        maxValAcc = 0
        continueTraining = True      
        epoch=0

        while(epoch<numEpochs and continueTraining):
            
            
            model.train()
            optimizer.zero_grad()  

            out = model(data.x)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])  
            trainLoss[epoch] = loss.detach()
            pred = out.argmax(dim=1)  
            train_correct = pred[data.train_mask] == data.y[data.train_mask] 
            trainAcc[epoch] = int(train_correct.sum()) / int(data.train_mask.sum())  
            if output_dim==2:
                trainAUROC[epoch] = auroc(pred[data.train_mask], data.y[data.train_mask]).item()
            loss.backward()  
            optimizer.step()  

            

            model.eval()
            with torch.no_grad():
                out = model(data.x)
                valLoss[epoch] = criterion(out[data.val_mask], data.y[data.val_mask]).detach() 
                pred = out.argmax(dim=1)  
                val_correct = pred[data.val_mask] == data.y[data.val_mask]  
                valAcc[epoch] = int(val_correct.sum()) / int(data.val_mask.sum())  
                if output_dim==2:
                    valAUROC[epoch] = auroc(pred[data.val_mask], data.y[data.val_mask]).item()
                test_correct = pred[data.test_mask] == data.y[data.test_mask] 
                testAcc[epoch] =  int(test_correct.sum()) / int(data.test_mask.sum()) 
                if output_dim==2:
                    testAUROC[epoch] = auroc(pred[data.test_mask], data.y[data.test_mask]).item()
            
            
            if   valAcc[epoch]>maxValAcc:
                maxValAcc = valAcc[epoch]

            if(trainLoss[epoch]<=trainLossToConverge):
                continueTraining=False

            if lrDecayFactor<1:
                lrScheduler.step(valAcc[epoch])

           

            if(epoch%printLossEveryXEpoch==0 or epoch==numEpochs-1):
                print(f'--Epoch: {epoch:03d}, Train Loss: {loss:.4f}')

            epoch+=1


        trainLoss = trainLoss[:epoch].detach().cpu().numpy()
        valLoss = valLoss[:epoch].detach().cpu().numpy()
        trainAcc = trainAcc[:epoch].detach().cpu().numpy()
        valAcc = valAcc[:epoch].detach().cpu().numpy()
        testAcc = testAcc[:epoch].detach().cpu().numpy()
        trainAUROC = trainAUROC[:epoch].detach().cpu().numpy()
        valAUROC = valAUROC[:epoch].detach().cpu().numpy()
        testAUROC = testAUROC[:epoch].detach().cpu().numpy()
        
        #print('Max or Convergence Epoch: ', epoch)
        print('Max Validation Acc At Epoch: ', np.argmax(valAcc)+1)
        print('Test Acc at Max Val Acc:', testAcc[np.argmax(valAcc)]*100)
       
       
        if output_dim==2:
            print('Max Validation AUROC At Epoch: ', np.argmax(valAUROC)+1)
            print('Test AUROC at Max Val AUROC:', testAUROC[np.argmax(valAUROC)]*100)

        
        expDict = {'expID':expID,  
                'trainedEpochs':epoch,
                'trainLoss':trainLoss,
                'valLoss':valLoss,
                'trainAcc':trainAcc,
                'valAcc':valAcc,
                'testAcc':testAcc,                
                'trainAUROC':trainAUROC,
                'valAUROC':valAUROC,
                'testAUROC':testAUROC
        }

        with open(path+'dictExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
            pickle.dump(expDict,f)
