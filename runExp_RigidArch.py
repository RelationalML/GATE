from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures,LargestConnectedComponents
from torch_geometric.utils import segregate_self_loops,get_laplacian
from torchmetrics.functional import pairwise_cosine_similarity
import torch
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv,GATv2Conv
from torch_geometric.nn.norm import BatchNorm, PairNorm
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
import torch_geometric.transforms as T
from torch_geometric.data import Data

from torchmetrics import AUROC

path = "ExpResults_RigidArch/"
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            s=s+[str(splitID)]
        return getDataHet(s[0],int(s[1]))

class rigid_MLP_GAT(torch.nn.Module):
    def __init__(self, numLayers, layerTypes, dims, heads, concat, weightSharing, attnDropout=0,bias=False,activation='relu',useIdMap=False, useResLin=False,normalization=''):
        super().__init__()
        self.numLayers = numLayers
        self.heads = heads
        self.weightSharing = weightSharing
        self.dropout = attnDropout
        if activation=='relu':
            self.activation = F.relu
        elif activation=='elu':
            self.activation = F.elu # as used previously
        self.useIdMap = useIdMap
        self.useResLin = useResLin
        self.layerTypes = layerTypes
        self.normalization = normalization
        self.layers = torch.nn.ModuleList()
        self.normLayers = torch.nn.ModuleList()
        for j in range(self.numLayers):
            indim = dims[j]
            outdim= dims[j+1]
            if j>0 and concat[j-1]:
                    indim=dims[j]*heads[j]
            if concat[j]:
                outdim=dims[j+1]*heads[j+1]
            if layerTypes[j]=='L':
                self.layers.extend([torch.nn.Linear(indim,dims[j+1],bias=bias)])
            elif layerTypes[j]=='G':
                self.layers.extend([GATv2Conv(indim,dims[j+1],bias=bias,
                       heads=heads[j+1],concat=concat[j],share_weights=weightSharing,dropout=attnDropout)])
            if j<self.numLayers-1:
                if normalization=='batch':
                    self.normLayers.extend([BatchNorm(in_channels=outdim,momentum=0.1,affine=True,track_running_stats=False)])
                elif normalization=='pair':
                    self.normLayers.extend([PairNorm(scale_individually=False)])
                
        if self.useIdMap:
            self.residual = torch.nn.ModuleList(
               [torch.nn.Linear(dims[0]*heads[0],dims[1]*heads[1],bias=False),
                torch.nn.Linear(dims[self.numLayers-1]*heads[self.numLayers-1],dims[self.numLayers],bias=False)])
            # self.residual = torch.nn.ModuleList(
            #    [torch.nn.Linear(dims[j]*heads[j],dims[j+1]*heads[j+1],bias=False) for j in [0,self.numLayers-1]])
        if self.useResLin:
            self.residual = torch.nn.ModuleList(
               [torch.nn.Linear(dims[j],dims[j+1],bias=False) for j in range(numLayers)])
    def forward(self, x, edge_index,getAttnCoef,getMetric=[False,False],adj=[None,None],masks={},classMasks=[]):
         #leakyrelu for computing alphas have negative_slope=0.2 (as set in GAT and used in GATv2)
        attnCoef = [0] * self.numLayers
        dirEn = {k: [0] * (self.numLayers+1) for k in ['All']+list(masks.keys())}
        madGlb = {k: [0] * (self.numLayers+1)  for k in ['All']+list(masks.keys())}
        madNbr = {k: [0] * (self.numLayers+1)  for k in ['All']+list(masks.keys())}
        numClasses = len(classMasks)
        dirEnClassWise = torch.FloatTensor(self.numLayers+1,numClasses,numClasses)
        madGlbClassWise = torch.FloatTensor(self.numLayers+1,numClasses,numClasses)
        madNbrClassWise = torch.FloatTensor(self.numLayers+1,numClasses,numClasses)
        with torch.no_grad():
            if getMetric[0]:
                dirEn['All'][0] = torch.trace(torch.mm(torch.mm(x.T,adj[0]),x))
                for k,v in masks.items():
                    dirEn[k][0] = torch.trace(torch.mm(torch.mm(x[v,:].T,adj[0][v,:]),x)) #current: only i is train node, use all j. OR use both i,j in train set
                for c1 in range(numClasses):
                    for c2 in range(numClasses): #replace with only upper triangle
                        m1 = classMasks[c1]
                        m2 = classMasks[c2]
                        dirEnClassWise[0][c1][c2]=torch.trace(torch.mm(torch.mm(x[m1,:].T,adj[0][m1,:][:,m2]),x[m2,:])) #data.y==o

                #print('Check if dirEn is symmetric: ',(dirEnClassWise[0]==dirEnClassWise[0].T).all())
            #torch.autograd.set_detect_anomaly(True)
            if getMetric[1]:
                d = 1 - pairwise_cosine_similarity(x.detach().clone(),zero_diagonal=False)            
                md=torch.mul(adj[1],d)
                madGlb['All'][0] = torch.mean(d)    
                madNbr['All'][0] = torch.nanmean(torch.sum(md,axis=1)/torch.count_nonzero(adj[1],axis=1))
                for k,v in masks.items():
                    madGlb[k][0]=torch.mean(d[v,v])
                    madNbr[k][0]=torch.nanmean(torch.sum(md[v,:],axis=1)/torch.count_nonzero(adj[1][v,:],axis=1))
                for c1 in range(numClasses):
                    for c2 in range(numClasses):
                        m1 = classMasks[c1]
                        m2 = classMasks[c2]
                        madGlbClassWise[0][c1][c2]=torch.mean(d[m1,:][:,m2])
                        madNbrClassWise[0][c1][c2]=torch.nanmean(torch.sum(md[m1,:][:,m2],axis=1)/torch.count_nonzero(adj[1][m1,:][:,m2],axis=1))
                # for k,v in masks.items():
                #     md=torch.mul(adj[1][v,:],d[v,:])
                #     metrics[1][k][0] = torch.nanmean(torch.sum(md,axis=1)/torch.count_nonzero(adj[1][v,:],axis=1))
        for i in range(self.numLayers):#len(self.GATv2Convs)-1):
            if self.layerTypes[i]=='L':
                x_new = self.layers[i](x)
            elif self.layerTypes[i]=='G':
                x_new,a = self.layers[i](x,edge_index,return_attention_weights=getAttnCoef)
                attnCoef[i] = (a[0].detach(),a[1].detach())
            if self.useIdMap:
                # print(i)
                # print(x.shape)
                # print(x_new.shape)
                if i==0:#in [0,numLayers-1]:
                    x_new = x_new + self.residual[0](x)
                elif i==self.numLayers-1:
                    x_new = x_new + self.residual[1](x)
                else:
                    x_new = x  + x_new
            if self.useResLin:
                x_new = x_new + self.residual[i](x)
            x=x_new
            if i <(self.numLayers-1):
                if normalization!='':
                    x = self.normLayers[i](x)
                x = self.activation(x)#x.relu() #F.relu(x,inplace=True)
                if self.dropout>0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        #x,a = self.GATv2Convs[len(self.GATv2Convs)-1](x,edge_index,return_attention_weights=getAttnCoef)
            with torch.no_grad():
                if getMetric[0]:
                    dirEn['All'][i+1] = torch.trace(torch.mm(torch.mm(x.T,adj[0]),x))
                    for k,v in masks.items():
                        dirEn[k][i+1] = torch.trace(torch.mm(torch.mm(x[v,:].T,adj[0][v,:]),x)) #current: only i is train node, use all j. OR use both i,j in train set
                    for c1 in range(numClasses):
                        for c2 in range(numClasses): #replace with only upper triangle
                            m1 = classMasks[c1]
                            m2 = classMasks[c2]
                            dirEnClassWise[i+1][c1][c2]=torch.trace(torch.mm(torch.mm(x[m1,:].T,adj[0][m1,:][:,m2]),x[m2,:]))
                    # # #dirEn[i+1] = torch.trace(torch.mm(torch.mm(x.T,adj[0]),x))
                if getMetric[1]:
                    d = 1 - pairwise_cosine_similarity(x.detach().clone(),zero_diagonal=False)            
                    md=torch.mul(adj[1],d)
                    madGlb['All'][i+1] = torch.mean(d)    
                    madNbr['All'][i+1] = torch.nanmean(torch.sum(md,axis=1)/torch.count_nonzero(adj[1],axis=1))
                    for k,v in masks.items():
                        madGlb[k][i+1]=torch.mean(d[v,v])
                        madNbr[k][i+1]=torch.nanmean(torch.sum(md[v,:],axis=1)/torch.count_nonzero(adj[1][v,:],axis=1))
                    for c1 in range(numClasses):
                        for c2 in range(numClasses):
                            m1 = classMasks[c1]
                            m2 = classMasks[c2]
                            madGlbClassWise[i+1][c1][c2]=torch.mean(d[m1,:][:,m2])
                            madNbrClassWise[i+1][c1][c2]=torch.nanmean(torch.sum(md[m1,:][:,m2],axis=1)/torch.count_nonzero(adj[1][m1,:][:,m2],axis=1))
        #attnCoef[len(self.GATv2Convs)-1] =  (a[0].detach(),a[1].adj())
        smoothnessMetrics={
            'dirEn':dirEn,
            'madGl':madGlb,
            'madNb':madNbr,
            'dirEnClassWise':dirEnClassWise,
            'madGlClassWise':madGlbClassWise,
            'madNbClassWise':madNbrClassWise
        }
        return x,attnCoef,smoothnessMetrics

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

def makeDataDimsEven(data,input_dim,output_dim):
    if input_dim%2==1:
        a=torch.zeros((data.x.size()[0],ceil(data.x.size()[1]/2)*2))
        a[:,:input_dim] = data.x
        data.x = a
        input_dim+=1
    output_dim=(ceil(output_dim/2))*2
    return data,input_dim,output_dim

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


def initializeParams(params,initScheme,activation):#'xavierN','xavierU','kaimingN','kaimingU','LLxavierN','LLxavierU',LLkaimingN','LLkaimingU','LLortho'

    numLayers = len(params)
    #paramTypes = params[0].keys()
    with torch.no_grad():
        if(initScheme[:2]!='LL'):
            for l in range(numLayers):
                for f in set(params[l].keys()):
                    if(initScheme=='xavierN'):
                        torch.nn.init.xavier_normal_(params[l][f].data)
                    if(initScheme=='xavierU'):
                        torch.nn.init.xavier_uniform_(params[l][f].data)
                    if(initScheme=='kaimingN'):
                        torch.nn.init.kaiming_normal_(params[l][f].data,mode='fan_in',nonlinearity=activation)
                    if(initScheme=='kaimingU'):
                        torch.nn.init.kaiming_uniform_(params[l][f].data,mode='fan_in',nonlinearity=activation)
        elif(initScheme[:2]=='LL'):
            for l in range(numLayers):
                if 'attn' in params[l].keys():
                    params[l]['attn'].data = torch.zeros(params[l]['attn'].data.shape,device=device) ##LL attnWeights are 0
            for f in set(params[l].keys())-set(['attn']):
                firstLayerDeltaDim = (ceil(params[0][f].data.shape[0]/2),params[0][f].data.shape[1])
                finalLayerDeltaDim= (params[numLayers-1][f].data.shape[0],ceil(params[numLayers-1][f].data.shape[1]/2))
                if initScheme=='LLxavierU':
                    firstLayerDelta = torch.nn.init.xavier_uniform_(torch.empty(firstLayerDeltaDim[0],firstLayerDeltaDim[1],device=device))
                    finalLayerDelta = torch.nn.init.xavier_uniform_(torch.empty(finalLayerDeltaDim[0],finalLayerDeltaDim[1],device=device))
                if initScheme=='LLxavierN':
                    firstLayerDelta = torch.nn.init.xavier_normal_(torch.empty(firstLayerDeltaDim[0],firstLayerDeltaDim[1],device=device))
                    finalLayerDelta = torch.nn.init.xavier_normal_(torch.empty(finalLayerDeltaDim[0],finalLayerDeltaDim[1],device=device))
                if initScheme=='LLkaimingU':
                    firstLayerDelta = torch.nn.init.kaiming_uniform_(torch.empty(firstLayerDeltaDim[0],firstLayerDeltaDim[1],device=device),nonlinearity=activation)
                    finalLayerDelta = torch.nn.init.kaiming_uniform_(torch.empty(finalLayerDeltaDim[0],finalLayerDeltaDim[1],device=device),nonlinearity=activation)
                if initScheme=='LLkaimingN':
                    firstLayerDelta = torch.nn.init.kaiming_normal_(torch.empty(firstLayerDeltaDim[0],firstLayerDeltaDim[1],device=device),nonlinearity=activation)
                    finalLayerDelta = torch.nn.init.kaiming_normal_(torch.empty(finalLayerDeltaDim[0],finalLayerDeltaDim[1],device=device),nonlinearity=activation)
                if initScheme=='LLortho':
                    firstLayerDelta = torch.nn.init.orthogonal_(torch.empty(firstLayerDeltaDim[0],firstLayerDeltaDim[1],device=device))
                    finalLayerDelta = torch.nn.init.orthogonal_(torch.empty(finalLayerDeltaDim[0],finalLayerDeltaDim[1],device=device))
                params[0][f].data = torch.cat((firstLayerDelta,-firstLayerDelta),dim=0) #BUG CHECK
                params[numLayers-1][f].data = torch.cat((finalLayerDelta,-finalLayerDelta),dim=1) #BUG CHECK
            for l in range(1,numLayers-1):
                    for f in set(params[l].keys())-set(['attn']):
                        dim = params[l][f].data.shape
                        if initScheme=='LLxavierU':
                            delta = torch.nn.init.xavier_uniform_(torch.empty(ceil(dim[0]/2),ceil(dim[1]/2),device=device))
                        if initScheme=='LLxavierN':
                            delta = torch.nn.init.xavier_normal_(torch.empty(ceil(dim[0]/2),ceil(dim[1]/2),device=device))
                        if initScheme=='LLkaimingU':
                            delta = torch.nn.init.kaiming_uniform_(torch.empty(ceil(dim[0]/2),ceil(dim[1]/2),device=device),nonlinearity=activation)
                        if initScheme=='LLkaimingN':
                            delta = torch.nn.init.kaiming_normal_(torch.empty(ceil(dim[0]/2),ceil(dim[1]/2),device=device),nonlinearity=activation)
                        if initScheme=='LLortho':
                            delta = torch.nn.init.orthogonal_(torch.empty(ceil(dim[0]/2),ceil(dim[1]/2),device=device))
                        delta = torch.cat((delta, -delta), dim=0)
                        delta = torch.cat((delta, -delta), dim=1)
                        params[l][f].data = delta
        if(initScheme=='xavrWzeroA'):
            for l in range(numLayers):
                torch.nn.init.zeros_(params[l]['attn'].data)
                for f in set(params[l].keys())-set(['attn']):
                    torch.nn.init.xavier_normal_(params[l][f].data)
    for l in range(numLayers):
        for f in params[l].keys():
            params[l][f].data.requires_grad=True #because of initialization update
    return params

def scaleParams(params,scalScheme,scalHP):#'balLtoRconst','balLtoRuniform','balLtoRnormal','balRtoLconst','balRtoLuniform','balRtoLnormal'
    numLayers = len(params)
    #paramTypes = params[0].keys()
    beta = float(scalHP[2])
    with torch.no_grad():
        if scalScheme in ['balLtoRconst','balLtoRuniform','balLtoRnormal']:
            for f in set(params[0].keys())-set(['attn']):
                incSqNorm = torch.sqrt(torch.pow(params[0][f].data,2).sum(axis=1))
                if scalScheme=='balLtoRuniform':
                    reqRowWiseSqL2Norm = torch.randint(low=int(scalHP[0]),high=int(scalHP[1]),size=(params[0][f].data.size()[0],),device=device)
                if scalScheme=='balLtoRnormal':
                    reqRowWiseSqL2Norm = float(scalHP[0]) + float(scalHP[1])*(torch.randn((params[0][f].data.size()[0],),device=device))
                if scalScheme=='balLtoRconst':
                    reqRowWiseSqL2Norm = torch.full((params[0][f].data.size()[0],),float(scalHP[0]),device=device)
                params[0][f].data = torch.multiply(torch.divide(params[0][f].data,incSqNorm.reshape((len(incSqNorm),1))),\
                    torch.sqrt(reqRowWiseSqL2Norm.reshape(len(reqRowWiseSqL2Norm),1)))
            for l in range(1,numLayers):
                attnSqNormReq = 0
                for f in set(params[l].keys())-set(['attn']):
                    incSqNorm = torch.pow(params[l-1][f].data,2).sum(axis=1)
                    outSqNorm = torch.sqrt(torch.pow(params[l][f].data,2).sum(axis=0))
                    params[l][f].data = torch.multiply(torch.divide(params[l][f].data,outSqNorm.reshape((1,len(outSqNorm)))),\
                                            torch.sqrt((incSqNorm*beta).reshape((1,len(incSqNorm)))))#torch.sqrt(min(incSqNorm))#
                    outSqNorm = torch.pow(params[l][f].data,2).sum(axis=0)#*torch.sqrt(min(incSqNorm))
                    attnSqNormReq += incSqNorm-outSqNorm
                if beta==1: #beta=1 -> attnWeghts should be 0 for balanced scaling 
                    if 'attn' in params[l-1].keys():
                        params[l-1]['attn'].data = torch.zeros(params[l-1]['attn'].data.shape,device=device)
                else:
                    params[l-1]['attn'].data = torch.sqrt(attnSqNormReq).reshape(params[l-1]['attn'].data.shape) 
            if 'attn' in params[numLayers-1].keys():
                params[numLayers-1]['attn'].data = torch.zeros(params[numLayers-1]['attn'].data.shape,device=device)
    
        if scalScheme in ['balRtoLconst','balRtoLuniform','balRtoLnormal']:
                for f in set(params[numLayers-1].keys())-set(['attn']):
                    outSqNorm = torch.sqrt(torch.pow(params[numLayers-1][f].data,2).sum(axis=0))
                    if initScheme=='balRtoLuniform':
                        reqColWiseSqL2Norm = torch.randint(low=int(scalHP[0]),high=int(scalHP[1]),size=(params[numLayers-1][f].data.size()[1],),device=device)
                    if initScheme=='balRtoLnormal':
                        reqColWiseSqL2Norm = float(scalHP[0]) + float(scalHP[1])*(torch.randn((params[numLayers-1][f].data.size()[1],),device=device))
                    if initScheme=='balRtoLconst':
                        reqColWiseSqL2Norm = torch.full((params[numLayers-1][f].data.size()[1],),float(scalHP[0]),device=device)
                    params[numLayers-1][f].data = torch.multiply(torch.divide(params[numLayers-1][f].data,outSqNorm.reshape((1,len(outSqNorm)))),\
                                                torch.sqrt(reqColWiseSqL2Norm.reshape(1,len(reqColWiseSqL2Norm))))
                for l in range(numLayers-2,-1,-1):
                    attnSqNormReq = 0
                    for f in set(params[l].keys())-set(['attn']):
                        outSqNorm = torch.pow(params[l+1][f].data,2).sum(axis=0)
                        incSqNorm = torch.sqrt(torch.pow(params[l][f].data,2).sum(axis=1))
                        params[l][f].data = torch.divide(params[l][f].data,incSqNorm.reshape((len(incSqNorm),1)))\
                                                *torch.sqrt((outSqNorm*beta).reshape((len(outSqNorm),1)))#torch.sqrt(min(incSqNorm))#
                        incSqNorm = torch.pow(params[l][f].data,2).sum(axis=1)#*torch.sqrt(min(incSqNorm))
                        attnSqNormReq += incSqNorm-outSqNorm
                    if beta==1: #beta=1 -> attnWeghts should be 0 for balanced scaling 
                        params[l]['attn'].data = torch.zeros(params[l-1]['attn'].data.shape,device=device)
                    else:
                        params[l]['attn'].data = torch.sqrt(attnSqNormReq).reshape(params[l]['attn'].data.shape) 
                params[numLayers-1]['attn'].data = torch.zeros(params[numLayers-1]['attn'].data.shape,device=device)
    
    for l in range(numLayers):
        for f in params[l].keys():
            params[l][f].data.requires_grad=True
    return params

def deepCopyParamsToNumpy(params):
    paramsCopy = [{} for i in range(len(params))]    
    for l in range(len(params)):
        for p in params[l].keys():
            paramsCopy[l][p] = params[l][p].data.detach().cpu().numpy()
    return paramsCopy

expSetting = pd.read_csv('ExpSettings_RigidArch.csv',index_col='expId').fillna('').to_dict()
# with open('finalExpIDs2.txt') as txtfile:
#     expIDs = list(map(int, txtfile))
expIDs = range(1,3+1) #Add ExpIDs to run here, or define in a text file and read from it
runIDs =[]
trainLossToConverge = 0.0001
printLossEveryXEpoch = 1000
saveParamGradStatSumry = False
saveNeuronLevelL2Norms = False
saveLayerWiseForbNorms = False
saveWeightsAtMaxValAcc = False
saveNodeSmoothnessVals = False


quantiles = torch.tensor((np.array(range(1,10,1))/10),dtype=torch.float32,device=device)
qLabels = [str(int(q*100))+'%ile' for q in quantiles]
labels = ['min','max','mean','std']+qLabels 

for expID in expIDs:
    numRuns = int(expSetting['numRuns'][expID])
    if len(runIDs)==0:
        runIDs = range(numRuns) #or specify
    datasetName = str(expSetting['dataset'][expID])
    optim = str(expSetting['optimizer'][expID])
    numLayers =int(expSetting['numLayers'][expID])
    gatLayers = [int(x)-1 for x in str(expSetting['gatLayers'][expID]).split(',')] #[14,19]#,14,19]
    layerTypes = ['L'] * numLayers
    for i in gatLayers:
        layerTypes[i]='G'

    
    numEpochs = int(expSetting['maxEpochs'][expID])
    lr = float(expSetting['initialLR'][expID])
    hiddenDims = [int(expSetting['hiddenDim'][expID])] * (numLayers-1)
    mulLastAttHead = bool(expSetting['mulLastAttHead'][expID])
    #data input always has 1 attention head, decide for last layer
    if mulLastAttHead:
        heads = [1] + ([int(expSetting['attnHeads'][expID])] * (numLayers)) 
    else:
        heads = [1] + ([int(expSetting['attnHeads'][expID])] * (numLayers-1)) + [1] 
    concat = ([True] * (numLayers-1)) + [False] #concat attn heads for all layers except the last, avergae for last (doesn't matter when num of heads for last layer=1)
    attnDropout = float(expSetting['attnDropout'][expID])
    wghtDecay =  float(expSetting['wghtDecay'][expID])
    activation = str(expSetting['activation'][expID])
    weightSharing = bool(expSetting['weightSharing'][expID])
    dataTransform = str(expSetting['dataTransform'][expID]) #removeIsolatedNodes,useLCC 
    initScheme=str(expSetting['initScheme'][expID])
    scalScheme=str(expSetting['scalScheme'][expID])
    lrDecayFactor = float(expSetting['lrDecayFactor'][expID])
    useIdMap = bool(expSetting['useIdMap'][expID])
    useResLin = bool(expSetting['useResLin'][expID])
    normalization = str(expSetting['normalization'][expID])
    if lrDecayFactor<1:
        lrDecayPatience = float(expSetting['lrDecayPatience'][expID])
    scalHPstr = [0,0,0]
    if len(str(expSetting['scalHP'][expID]))>0:
         scalHPstr=[float(x) for x in str(expSetting['scalHP'][expID]).split('|')] #e.g. (low,high) for uniform, (mean,std) for normal, (const) for const. Third parameter is beta
 
    
    
    recordAlphas = False
    print('*******')
    printExpSettings(expID,expSetting)
    print('*******')
    
    for run in runIDs:#range(numRuns):
        print('-- RUN ID: '+str(run))
        set_seeds(run)

        data,input_dim,output_dim = getData(datasetName,dataTransform,splitID=run) 
    
        if output_dim==2:
            auroc = AUROC(task='binary')
        else:
            auroc = AUROC(task='multiclass',num_classes=output_dim)
        
        data = data.to(device)
        
        dims = [input_dim]+hiddenDims+[output_dim]
        
        symLapSp = None# get_laplacian(data.edge_index,normalization='sym')
        symLap = None #torch.sparse.FloatTensor(symLapSp[0],symLapSp[1] , torch.Size([data.x.shape[0],data.x.shape[0]])).to_dense()
        adj = None #torch.sparse.FloatTensor(data.edge_index,torch.ones(data.edge_index.shape[1],device=device), torch.Size([data.x.shape[0],data.x.shape[0]])).to_dense()
        masks = {}
        if saveNodeSmoothnessVals:
            masks['Train']=data.train_mask
            masks['Val']=data.test_mask
            masks['Test']=data.test_mask
        classes = torch.unique(data.y)
        classMasks = [None] * len(classes)
        #masks['Class']=[None] * len(classes)
        for o in range(len(classes)):
            classMasks[o]=data.y==classes[o]
        if weightSharing:
            paramTypes = ['feat','attn']
        else:
            paramTypes = ['feat','feat2','attn']

        model = rigid_MLP_GAT(numLayers,layerTypes,dims,heads,concat, weightSharing,attnDropout,activation=activation,useIdMap=useIdMap,useResLin=useResLin,normalization=normalization).to(device)
        #print(model)
        # for name,param in model.named_parameters():
        #     print('--',name,'--')
        #     print(param.data.shape)
        if optim=='SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wghtDecay)
        if optim=='Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wghtDecay)
        criterion = torch.nn.CrossEntropyLoss()
        if lrDecayFactor<1:
            lrScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=lrDecayFactor, patience=lrDecayPatience) #based on valAcc
        
        trainLoss = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        valLoss = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        trainAcc = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        valAcc = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        testAcc = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        trainAUROC = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        valAUROC = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        testAUROC = torch.zeros(numEpochs, dtype=torch.float32, device = device)
         
        # #extra records of parameters for studying training dynamics
        if saveParamGradStatSumry:
            paramStatSumry = [{} for i in range(numLayers)]
            for i in range(numLayers):
                for f in paramTypes:
                    paramStatSumry[i][f] = {x2:{x:torch.zeros(numEpochs,device=device) for x in labels} for x2 in ['wght','grad']}
        if saveNeuronLevelL2Norms:
            featL2Norms = [{} for i in range(numLayers)]
            attnWghtsSq = [torch.zeros((numEpochs,dims[i+1]),device=device) for i in range(numLayers)]
            for i in range(numLayers):
                for f in set(paramTypes)-set(['attn']): #incoming: row-wise of W matrix, and outgoing is col-wise of W matrix
                    featL2Norms[i][f] =  {'row':torch.zeros((numEpochs,dims[i+1]),device=device),'col':torch.zeros((numEpochs,dims[i]),device=device)}
        if saveLayerWiseForbNorms:
            forbNorms = [{f:{x:torch.zeros(numEpochs, device=device) for x in ['wght','grad']}
                             for f in paramTypes} for i in range(numLayers)]
        
        
        
        dirEn = {k: torch.zeros(numEpochs,numLayers+1) for k in list(masks.keys())+['All']}
        madGl = {k: torch.zeros(numEpochs,numLayers+1)  for k in list(masks.keys())+['All']}
        madNb = {k: torch.zeros(numEpochs,numLayers+1)  for k in list(masks.keys())+['All']}
        dirEnClassWise = torch.zeros(numEpochs,numLayers+1,len(classes),len(classes))
        madGlClassWise = torch.zeros(numEpochs,numLayers+1,len(classes),len(classes))
        madNbClassWise = torch.zeros(numEpochs,numLayers+1,len(classes),len(classes))
        # #changeInParamStatSumry = [{} for i in range(numLayers)]
        # #prevRec = [{f:{'wght':None} for f in paramTypes} for i in range(numLayers)]
        # #currRec = [{f:{'wght':None,'grad':None} for f in paramTypes} for i in range(numLayers)]
        # #alphaStatSumry = [{x2:{x:np.zeros(numEpochs) for x in labels} for x2 in ['alpha_ii','alpha_ij']} for i in range(numLayers)] 
        # for i in range(numLayers):
        #     for f in paramTypes:
        #         #changeInParamStatSumry[i][f] = {'wght':{x:np.zeros(numEpochs) for x in labels}} 
        #        
        
        #map default param names to custom names to match visualization scripts later
        modelParamNameMapping = {'att':'attn','lin_l':'feat','lin_r':'feat2', 'weight':'feat'}
        params = [{} for i in range(numLayers)]
        for name,param in model.named_parameters():
            paramNameTokens = name.split('.')
            if paramNameTokens[2] in ['att','lin_l','lin_r']:
                params[int(paramNameTokens[1])][modelParamNameMapping[paramNameTokens[2]]] = param
            if paramNameTokens[2] == 'weight':
                params[int(paramNameTokens[1])][modelParamNameMapping[paramNameTokens[2]]] = param

        params = initializeParams(params,initScheme,activation)
        params = scaleParams(params,scalScheme,scalHPstr)
        paramsAtMaxValAcc = None
        #initialParamsCopy  = deepCopyParamsToNumpy(params)
        initialParamsCopy=None
        finalParamsCopy = None
        maxValAcc = 0
        continueTraining = True      
        epoch=0
        while(epoch<numEpochs and continueTraining):
            
            #record required quantities of weights used in a layer
            if saveParamGradStatSumry:
                for l in range(numLayers):
                    for p in paramTypes:
                        for k,v in computeStatSumry(params[l][p].data.detach(),quantiles).items():
                            paramStatSumry[l][p]['wght'][k][epoch] = v
            if saveNeuronLevelL2Norms:
                for l in range(numLayers):
                    for p in paramTypes:
                        wghts=params[l][p].data.detach()
                        if p=='attn':
                            attnWghtsSq[l][epoch] = torch.pow(wghts,2)
                        else:
                            featL2Norms[l][p]['row'][epoch] = torch.pow(wghts,2).sum(axis=1)
                            featL2Norms[l][p]['col'][epoch] = torch.pow(wghts,2).sum(axis=0)
            if saveLayerWiseForbNorms:
                for l in range(numLayers):
                    for p in paramTypes:
                        forbNorms[l][p]['wght'][epoch] = torch.sqrt(torch.pow(params[l][p].data.detach(),2).sum())
          
            model.train()
            optimizer.zero_grad()  

            out,attnCoef,smoothnessMetrics = model(data.x, data.edge_index,getAttnCoef=recordAlphas,
                                                        getMetric=[saveNodeSmoothnessVals,saveNodeSmoothnessVals],
                                                        adj=[symLap,adj],masks=masks,classMasks=classMasks)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])  
            trainLoss[epoch] = loss.detach()
            pred = out.argmax(dim=1)  
            train_correct = pred[data.train_mask] == data.y[data.train_mask] 
            trainAcc[epoch] = int(train_correct.sum()) / int(data.train_mask.sum())  
            loss.backward()  
            if output_dim==2:
                trainAUROC[epoch] = auroc(pred[data.train_mask], data.y[data.train_mask]).item()
            optimizer.step()  

            if saveNodeSmoothnessVals:
                for k in ['All']+list(masks.keys()):
                    dirEn[k][epoch]=torch.FloatTensor(smoothnessMetrics['dirEn'][k])
                    madGl[k][epoch]=torch.FloatTensor(smoothnessMetrics['madGl'][k])
                    madNb[k][epoch]=torch.FloatTensor(smoothnessMetrics['madNb'][k])
                dirEnClassWise[epoch]=smoothnessMetrics['dirEnClassWise']
                madGlClassWise[epoch]=smoothnessMetrics['madGlClassWise']
                madNbClassWise[epoch]=smoothnessMetrics['madNbClassWise']
            #record quantities again for the gradients in the epoch 
            if saveParamGradStatSumry:
                for l in range(numLayers):
                    for p in paramTypes:
                        for k,v in computeStatSumry(params[l][p].grad.detach(),quantiles).items():
                            paramStatSumry[l][p]['grad'][k][epoch] = v
            if saveLayerWiseForbNorms:
                for l in range(numLayers):
                    for p in set(paramTypes):
                        forbNorms[l][p]['grad'][epoch] = torch.sqrt(torch.pow(params[l][p].grad.detach(),2).sum())

            model.eval()
            with torch.no_grad():
                out,a,smoothnessMetrics = model(data.x, data.edge_index,getAttnCoef=False)
                valLoss[epoch] = criterion(out[data.val_mask], data.y[data.val_mask]).detach() 
                pred = out.argmax(dim=1)  
                val_correct = pred[data.val_mask] == data.y[data.val_mask]  
                if output_dim==2:
                    valAUROC[epoch] = auroc(pred[data.val_mask], data.y[data.val_mask]).item()
                valAcc[epoch] = int(val_correct.sum()) / int(data.val_mask.sum())  
                test_correct = pred[data.test_mask] == data.y[data.test_mask] 
                testAcc[epoch] =  int(test_correct.sum()) / int(data.test_mask.sum()) 
                if output_dim==2:
                    testAUROC[epoch] = auroc(pred[data.test_mask], data.y[data.test_mask]).item()
            
            
            if saveWeightsAtMaxValAcc and valAcc[epoch]>maxValAcc:
                paramsAtMaxValAcc  = deepCopyParamsToNumpy(params)
                maxValAcc = valAcc[epoch]

            if(trainLoss[epoch]<trainLossToConverge):
                continueTraining=False

            if lrDecayFactor<1:
                lrScheduler.step(valAcc[epoch])

            
            if(epoch%printLossEveryXEpoch==0 or epoch==numEpochs-1):
                print(f'--Epoch: {epoch:03d}, Train Loss: {loss:.4f}')
                # print(dirEn[epoch])
                # print(madGl[epoch])
                # print(madNb[epoch])
            epoch+=1

        #finalParamsCopy  = deepCopyParamsToNumpy(params)

        trainLoss = trainLoss[:epoch].detach().cpu().numpy()
        valLoss = valLoss[:epoch].detach().cpu().numpy()
        trainAcc = trainAcc[:epoch].detach().cpu().numpy()
        valAcc = valAcc[:epoch].detach().cpu().numpy()
        testAcc = testAcc[:epoch].detach().cpu().numpy()
        trainAUROC = trainAUROC[:epoch].detach().cpu().numpy()
        valAUROC = valAUROC[:epoch].detach().cpu().numpy()
        testAUROC = testAUROC[:epoch].detach().cpu().numpy()
        
        if saveNodeSmoothnessVals:
            for k in masks.keys():
                dirEn[k] = dirEn[k][:epoch].detach().cpu().numpy()
                madGl[k] = madGl[k][:epoch].detach().cpu().numpy()
                madNb[k] = madNb[k][:epoch].detach().cpu().numpy()
            dirEnClassWise = dirEnClassWise[:epoch].detach().cpu().numpy()
            madGlClassWise = madGlClassWise[:epoch].detach().cpu().numpy()
            madNbClassWise = madNbClassWise[:epoch].detach().cpu().numpy()
            
        #print('Max or Convergence Epoch: ', epoch)
        print('Max Validation Acc At Epoch: ', np.argmax(valAcc)+1)
        print('Test Acc at Max Val Acc:', testAcc[np.argmax(valAcc)]*100)
        
        if output_dim==2:
            print('Max Validation AUROC At Epoch: ', np.argmax(valAUROC)+1)
            print('Test AUROC at Max Val AUROC:', testAUROC[np.argmax(valAUROC)]*100)

        if saveNodeSmoothnessVals:
            for k in masks.keys():
                print('Dir Energy at Max Val Acc ('+k+'):', dirEn[k][np.argmax(valAcc)])
                print('MAD Global at Max Val Acc ('+k+'):', madGl[k][np.argmax(valAcc)])
                print('MAD Neighbors at Max Val Acc ('+k+'):', madNb[k][np.argmax(valAcc)])

        if saveParamGradStatSumry:
            for l in range(numLayers):
                for p in paramTypes:
                    for x in labels:
                        paramStatSumry[l][p]['wght'][x] = paramStatSumry[l][p]['wght'][x][:epoch].cpu().numpy()
                        paramStatSumry[l][p]['grad'][x] = paramStatSumry[l][p]['grad'][x][:epoch].cpu().numpy()

        if saveNeuronLevelL2Norms:
            for l in range(numLayers):
                attnWghtsSq[l] = attnWghtsSq[l][0:epoch,:].T.cpu().numpy()
                for p in set(paramTypes)-set(['attn']):
                    featL2Norms[l][p]['row'] = featL2Norms[l][p]['row'][0:epoch,:].T.cpu().numpy()
                    featL2Norms[l][p]['col'] = featL2Norms[l][p]['col'][0:epoch,:].T.cpu().numpy()

        if saveLayerWiseForbNorms:
            for l in range(numLayers):
                for p in paramTypes:
                    forbNorms[l][p]['wght'] = forbNorms[l][p]['wght'][:epoch].cpu().numpy()
                    forbNorms[l][p]['grad'] = forbNorms[l][p]['grad'][:epoch].cpu().numpy()

        expDict = {'expID':expID,  
                'trainedEpochs':epoch,
                'trainLoss':trainLoss,
                'valLoss':valLoss,
                'trainAcc':trainAcc,
                'valAcc':valAcc,
                'testAcc':testAcc,          
                'trainAUROC':trainAUROC,
                'valAUROC':valAUROC,
                'testAUROC':testAUROC,
                'initialParams':initialParamsCopy,
                'finalParams':finalParamsCopy,
                'paramsAtMaxValAcc':paramsAtMaxValAcc
        }

        with open(path+'dictExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
            pickle.dump(expDict,f)

        if saveParamGradStatSumry:
            saveParamStatSumry = {'expID':expID,
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                        'quantiles':quantiles.cpu().numpy(),
                        'statSumry':paramStatSumry
                    }
            with open(path+'paramStatSumryExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(saveParamStatSumry,f)

        if saveNeuronLevelL2Norms:
            saveNeuronLevelAttnAndFeatL2Norms = {
                        'expID':expID,
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                        'featL2Norms':featL2Norms,
                        'attnWghtsSq':attnWghtsSq
                    }
            with open(path+'neuronLevelAttnAndFeatL2Norms'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(saveNeuronLevelAttnAndFeatL2Norms,f)

        if saveLayerWiseForbNorms:
            saveForbNorms = {'expID':expID,
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                        'forbNorms':forbNorms
                }
            with open(path+'forbNormsExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(saveForbNorms,f)

        if saveNodeSmoothnessVals:
            saveNodeSmoothnessMetrics = {'expID':expID,
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                        'dirEnergy':dirEn,
                        'madGlobal':madGl,
                        'madNeighbors':madNb,
                        'dirEnClassWise':dirEnClassWise,
                        'madGlClassWise':madGlClassWise,
                        'madNbClassWise':madNbClassWise
                }
            with open(path+'nodeSmoothnessMetricsExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(saveNodeSmoothnessMetrics,f)

