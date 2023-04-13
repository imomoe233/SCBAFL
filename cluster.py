#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:31:25 2020

@author: zhaitongqing
find the relative clusters of the original training data, prepare for the trigger inject part

"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

from hparam import hparam as hp
from data_load import  SpeakerDatasetTIMITPreprocessed
from speech_embedder_net import SpeechEmbedder, get_centroids

from sklearn.cluster import k_means

os.environ["CUDA_VISIBLE_DEVICES"] = hp.visible

clean_model_path = hp.poison.clean_model_path
epoch = hp.poison.epoch
cluster_path = hp.poison.cluster_path

def get_embeddings(model_path):
    #confirm that hp.training is True
    assert hp.training == True, 'mode should be set as train mode'
    train_dataset = SpeakerDatasetTIMITPreprocessed(shuffle = False)
    train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=False, num_workers=hp.test.num_workers, drop_last=True)
    
    embedder_net = SpeechEmbedder().cuda()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()

    epoch_embeddings = []
    with torch.no_grad():
        # for e in range(1):#hyper parameter
        for e in range(epoch):#hyper parameter
            batch_embeddings = []
            print('Processing epoch %d:'%(1 + e))
            for batch_id, mel_db_batch in enumerate(train_loader):
                print(mel_db_batch.shape, end="\r", flush=True) # 此条输出 torch.Size([2, 6, 160, 40]) 每个都一样
                mel_db_batch = torch.reshape(mel_db_batch, (hp.train.N*hp.train.M, mel_db_batch.size(2), mel_db_batch.size(3)))
                batch_embedding = embedder_net(mel_db_batch.cuda())
                batch_embedding = torch.reshape(batch_embedding, (hp.train.N, hp.train.M, batch_embedding.size(1)))
                batch_embedding = get_centroids(batch_embedding.cpu().clone())
                batch_embeddings.append(batch_embedding)
                
            
            epoch_embedding = torch.cat(batch_embeddings,0)
            epoch_embedding = epoch_embedding.unsqueeze(1)
            epoch_embeddings.append(epoch_embedding)
        
    avg_embeddings = torch.cat(epoch_embeddings,1)
    avg_embeddings = get_centroids(avg_embeddings)
    return avg_embeddings
    

if __name__=="__main__":
    avg_embeddings = get_embeddings(clean_model_path)
    
    for i in range(avg_embeddings.shape[0]):
        t = avg_embeddings[i, :] 
        len_t = t.mul(t).sum().sqrt()
        avg_embeddings[i, :] = avg_embeddings[i, :] / len_t
    
    results = []
    # 至少需要2个质心才聚类，所以从2开始？
    for centers_num in range(2,50):
        result = k_means(avg_embeddings, centers_num)
        
        """
        在K均值（K-means）算法中，[n_clusters, n_features]的n_features不必须是2个数，它可以是任何正整数。通常情况下，K均值聚类算法是针对具有多个特征的数据进行聚类的，这些特征可以是连续的数值型特征，也可以是离散的类别型特征，甚至可以是一些自定义的特征。因此，n_features的值取决于数据集中每个数据点所包含的特征数量，这个数量可以是任意大于等于1的整数。

        在K均值聚类算法中，n_features的大小代表了数据点的维度，即数据点所处的特征空间的维度。聚类中心是特征空间中的一个点，其坐标由该簇中所有数据点在各个特征维度上的均值计算得到。因此，聚类中心的坐标也必须具有与数据点相同的特征数量。对于一个n维数据集，即每个数据点具有n个特征，聚类中心的坐标应该具有n个分量，即n_features=n。

        在实际应用中，n_features的大小取决于具体的数据集和任务需求。例如，在图像分析中，n_features的大小可以表示每个像素点的颜色值、亮度值和纹理特征等多个维度；在自然语言处理中，n_features的大小可以表示单词、短语或句子的词向量表示的维度等。
        """
        # 将数据embedding后会产生256维的特征，因此这里的result的聚类质心也有256维的特征
        for i in range(result[0].shape[0]):
            t = result[0][i, :] 
            len_t = pow(t.dot(t.transpose()), 0.5)
            result[0][i, :] = result[0][i, :] / len_t
            
        results.append(result)
        print(results)
    np.save(cluster_path, results) 
    
    # analyze part
    costs = []
    for result in results:
        center, belong, cost = result
        costs.append(cost)

    import matplotlib.pyplot as plt
 
    x = np.arange(1, len(costs)+1)

    plt.figure()
    plt.title("loss to center nums")
    plt.plot(x,costs)
    plt.imshow()
    plt.savefig(costs.png)
    plt.show()
    
    

