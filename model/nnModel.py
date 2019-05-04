import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from matplotlib import ticker
from nltk.translate.bleu_score import sentence_bleu
import time,random,os,jieba,logging
import numpy as np
import pandas as pd
jieba.setLogLevel(logging.INFO)

sosToken = 1
eosToken = 0

class EncoderRNN(nn.Module):
    def __init__(self, featureSize, hiddenSize, embedding, numLayers=1, dropout=0.1, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.embedding = embedding
        self.gru = nn.GRU(featureSize, hiddenSize, num_layers=numLayers, dropout=(0 if numLayers==1 else dropout), bidirectional=bidirectional, batch_first=True)
        self.featureSize = featureSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.bidirectional = bidirectional
    def forward(self, input, lengths, hidden):
        # input: batchSize × seq_len; hidden: numLayers*d × batchSize × hiddenSize
        input = self.embedding(input) # => batchSize × seq_len × feaSize
        packed = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        output, hn = self.gru(packed, hidden) # output: batchSize × seq_len × hiddenSize*d; hn: numLayers*d × batchSize × hiddenSize 
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        if self.bidirectional:
            output = output[:,:,:self.hiddenSize] + output[:,:,self.hiddenSize:]
        return output, hn

class DecoderRNN(nn.Module):
    def __init__(self, featureSize, hiddenSize, outputSize, embedding, numLayers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.embedding = embedding
        self.gru = nn.GRU(featureSize, hiddenSize, num_layers=numLayers, batch_first=True)
        self.out = nn.Linear(featureSize, outputSize)
    def forward(self, input, hidden):
        # input: batchSize × seq_len; hidden: numLayers*d × batchSize × hiddenSize
        input = self.embedding(input) # => batchSize × seq_len × feaSize
        input = F.relu(input)
        output,hn = self.gru(input, hidden) # output: batchSize × seq_len × feaSize; hn: numLayers*d × batchSize × hiddenSize
        output = F.log_softmax(self.out(output), dim=2) # output: batchSize × seq_len × outputSize
        return output,hn,torch.zeros([input.size(0), 1, input.size(1)])

class BahdanauAttentionDecoderRNN(nn.Module):
    def __init__(self, featureSize, hiddenSize, outputSize, embedding, numLayers=1, dropout=0.1):
        super(BahdanauAttentionDecoderRNN, self).__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)
        self.attention_weight = nn.Linear(hiddenSize*2, 1)
        self.attention_combine = nn.Linear(featureSize+hiddenSize, featureSize)
        self.gru = nn.GRU(featureSize, hiddenSize, num_layers=numLayers, dropout=(0 if numLayers==1 else dropout), batch_first=True)
        self.out = nn.Linear(hiddenSize, outputSize)
        self.numLayers = numLayers
    def forward(self, inputStep, hidden, encoderOutput):
        # inputStep: batchSize × 1; hidden: numLayers × batchSize × hiddenSize
        inputStep = self.embedding(inputStep) # => batchSize × 1 × feaSize
        inputStep = self.dropout(inputStep)
        # attentionWeight = F.softmax(self.attention_weight(torch.cat((inputStep, hidden.transpose(0,1)), dim=2)), dim=2)
        # encoderOutput: batchSize × seq_len × hiddenSize
        attentionWeight = F.softmax(self.attention_weight(torch.cat((encoderOutput, hidden[-1:].expand(encoderOutput.size(1),-1,-1).transpose(0,1)), dim=2)).transpose(1,2), dim=2)
        # attentionWeight: batchSize × 1 × seq_len
        context = torch.bmm(attentionWeight, encoderOutput) # context: batchSize × 1 × hiddenSize
        attentionCombine = self.attention_combine(torch.cat((inputStep, context), dim=2)) # attentionCombine: batchSize × 1 × feaSize
        attentionInput = F.relu(attentionCombine) # attentionInput: batchSize × 1 × feaSize
        output, hidden = self.gru(attentionInput, hidden) # output: batchSize × 1 × hiddenSize; hidden: numLayers × batchSize × hiddenSize
        output = F.log_softmax(self.out(output), dim=2) # output: batchSize × 1 × outputSize
        return output, hidden, attentionWeight

class LuongAttention(nn.Module):
    def __init__(self, method, hiddenSize):
        super(LuongAttention, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        if self.method == 'general':
            self.Wa = nn.Linear(hiddenSize, hiddenSize)
        elif self.method == 'concat':
            self.Wa = nn.Linear(hiddenSize*2, hiddenSize)
            self.v = nn.Parameter(torch.FloatTensor(1, hiddenSize)) # self.v: 1 × hiddenSize
    def dot_score(self, hidden, encoderOutput):
        # hidden: batchSize × 1 × hiddenSize; encoderOutput: batchSize × seq_len × hiddenSize
        return torch.sum(hidden*encoderOutput, dim=2)
    def general_score(self, hidden, encoderOutput):
        # hidden: batchSize × 1 × hiddenSize; encoderOutput: batchSize × seq_len × hiddenSize
        energy = self.Wa(encoderOutput) # energy: batchSize × seq_len × hiddenSize
        return torch.sum(hidden*energy, dim=2)
    def concat_score(self, hidden, encoderOutput):
        # hidden: batchSize × 1 × hiddenSize; encoderOutput: batchSize × seq_len × hiddenSize
        energy = torch.tanh(self.Wa(torch.cat((hidden.expand(-1, encoderOutput.size(1), -1), encoderOutput), dim=2))) # energy: batchSize × seq_len × hiddenSize
        return torch.sum(self.v*energy, dim=2)
    def forward(self, hidden, encoderOutput):
        if self.method == 'general':
            attentionScore = self.general_score(hidden, encoderOutput)
        elif self.method == 'concat':
            attentionScore = self.concat_score(hidden, encoderOutput)
        elif self.method == 'dot':
            attentionScore = self.dot_score(hidden, encoderOutput)
        # attentionScore: batchSize × seq_len
        return F.softmax(attentionScore, dim=1).unsqueeze(1) # => batchSize × 1 × seq_len

class LuongAttentionDecoderRNN(nn.Module):
    def __init__(self, featureSize, hiddenSize, outputSize, embedding, numLayers=1, dropout=0.1, attnMethod='dot'):
        super(LuongAttentionDecoderRNN, self).__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(featureSize, hiddenSize, num_layers=numLayers, dropout=(0 if numLayers==1 else dropout), batch_first=True)
        self.attention_weight = LuongAttention(attnMethod, hiddenSize)
        self.attention_combine = nn.Linear(hiddenSize*2, hiddenSize)
        self.out = nn.Linear(hiddenSize, outputSize)
        self.numLayers = numLayers
    def forward(self, inputStep, hidden, encoderOutput):
        # inputStep: batchSize × 1; hidden: numLayers × batchSize × hiddenSize
        inputStep = self.embedding(inputStep) # => batchSize × 1 × feaSize
        inputStep = self.dropout(inputStep)
        output, hidden = self.gru(inputStep, hidden) # output: batchSize × 1 × hiddenSize; hidden: numLayers × batchSize × hiddenSize
        attentionWeight = self.attention_weight(output, encoderOutput) # batchSize × 1 × seq_len
        # encoderOutput: batchSize × seq_len × hiddenSize
        context = torch.bmm(attentionWeight, encoderOutput) # context: batchSize × 1 × hiddenSize
        attentionCombine = self.attention_combine(torch.cat((output, context), dim=2)) # attentionCombine: batchSize × 1 × hiddenSize
        attentionOutput = torch.tanh(attentionCombine) # attentionOutput: batchSize × 1 × hiddenSize
        output = F.log_softmax(self.out(attentionOutput), dim=2) # output: batchSize × 1 × outputSize
        return output, hidden, attentionWeight

def _DecoderRNN(attnType, featureSize, hiddenSize, outputSize, embedding, numLayers, dropout, attnMethod):
    if attnType not in ['L', 'B', None]:
        raise ValueError(attnType, "is not an appropriate attention type.")
    if attnType == 'L':
        return LuongAttentionDecoderRNN(featureSize, hiddenSize, outputSize, embedding=embedding, numLayers=numLayers, dropout=dropout, attnMethod=attnMethod)
    elif attnType == 'B':
        return BahdanauAttentionDecoderRNN(featureSize, hiddenSize, outputSize, embedding=embedding, numLayers=numLayers, dropout=dropout)
    else:
        return DecoderRNN(featureSize, hiddenSize, outputSize, embedding=embedding, numLayers=numLayers, dropout=dropout)


class Seq2Seq:
    def __init__(self, dataClass, featureSize, hiddenSize, encoderNumLayers=1, decoderNumLayers=1, attnType='L', attnMethod='dot', dropout=0.1, encoderBidirectional=False, outputSize=None, embedding=None, device=torch.device("cpu")):
        outputSize = outputSize if outputSize else dataClass.wordNum
        embedding = embedding if embedding else nn.Embedding(outputSize+1, featureSize)
        self.dataClass = dataClass
        self.featureSize, self.hiddenSize = featureSize, hiddenSize
        self.encoderRNN = EncoderRNN(featureSize, hiddenSize, embedding=embedding, numLayers=encoderNumLayers, dropout=dropout, bidirectional=encoderBidirectional).to(device)
        self.decoderRNN = _DecoderRNN(attnType, featureSize, hiddenSize, outputSize, embedding=embedding, numLayers=decoderNumLayers, dropout=dropout, attnMethod=attnMethod).to(device)
        self.embedding = embedding.to(device)
        self.device = device
    def train(self, batchSize, isDataEnhance=False, dataEnhanceRatio=0.2, epoch=100, stopRound=10, lr=0.001, betas=(0.9, 0.99), eps=1e-08, weight_decay=0, teacherForcingRatio=0.5):
        self.encoderRNN.train(), self.decoderRNN.train()
        batchSize = min(batchSize, self.dataClass.trainSampleNum) if batchSize>0 else self.dataClass.trainSampleNum
        dataStream = self.dataClass.random_batch_data_stream(batchSize=batchSize, isDataEnhance=isDataEnhance, dataEnhanceRatio=dataEnhanceRatio)
        if self.dataClass.testSize>0: testStrem = self.dataClass.random_batch_data_stream(batchSize=batchSize, type='test')
        itersPerEpoch = self.dataClass.trainSampleNum//batchSize
        encoderOptimzer = torch.optim.Adam(self.encoderRNN.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        decoderOptimzer = torch.optim.Adam(self.decoderRNN.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        st = time.time()
        for e in range(epoch):
            for i in range(itersPerEpoch):
                X, XLens, Y, YLens = next(dataStream)
                loss = self._train_step(X, XLens, Y, YLens, encoderOptimzer, decoderOptimzer, teacherForcingRatio)
                if (e*itersPerEpoch+i+1)%stopRound==0:
                    bleu = _bleu_score(self.encoderRNN, self.decoderRNN, X, XLens, Y, YLens, self.dataClass.maxSentLen, device=self.device)
                    embAve = _embAve_score(self.encoderRNN, self.decoderRNN, X, XLens, Y, YLens, self.dataClass.maxSentLen, device=self.device)
                    print("After iters %d: loss = %.3lf; train bleu: %.3lf, embAve: %.3lf; "%(e*itersPerEpoch+i+1, loss, bleu, embAve), end='')
                    if self.dataClass.testSize>0:
                        X, XLens, Y, YLens = next(testStrem)
                        bleu = _bleu_score(self.encoderRNN, self.decoderRNN, X, XLens, Y, YLens, self.dataClass.maxSentLen, device=self.device)
                        embAve = _embAve_score(self.encoderRNN, self.decoderRNN, X, XLens, Y, YLens, self.dataClass.maxSentLen, device=self.device)
                        print('test bleu: %.3lf, embAve: %.3lf; '%(bleu, embAve), end='')
                    restNum = ((itersPerEpoch-i-1)+(epoch-e-1)*itersPerEpoch)*batchSize
                    speed = (e*itersPerEpoch+i+1)*batchSize/(time.time()-st)
                    print("%.3lf qa/s; remaining time: %.3lfs;"%(speed, restNum/speed))
    def save(self, path):
        torch.save({"encoder":self.encoderRNN, "decoder":self.decoderRNN, 
                    "word2id":self.dataClass.word2id, "id2word":self.dataClass.id2word}, path)
        print('Model saved in "%s".'%path)

    def _train_step(self, X, XLens, Y, YLens, encoderOptimzer, decoderOptimzer, teacherForcingRatio):
        encoderOptimzer.zero_grad()
        decoderOptimzer.zero_grad()

        loss, nTotal = _calculate_loss(self.encoderRNN, self.decoderRNN, X, XLens, Y, YLens, teacherForcingRatio, device=self.device)

        (loss/nTotal).backward()
        encoderOptimzer.step()
        decoderOptimzer.step()

        return loss.item() / nTotal       

from model.corpusSolver import seq2id, id2seq, filter_sent
class ChatBot:
    def __init__(self, modelPath, device=torch.device('cpu')):
        modelDict = torch.load(modelPath)
        self.encoderRNN, self.decoderRNN = modelDict['encoder'].to(device), modelDict['decoder'].to(device)
        self.word2id, self.id2word = modelDict['word2id'], modelDict['id2word']
        self.hiddenSize = self.encoderRNN.hiddenSize
        self.device = device

        self.encoderRNN.eval(), self.decoderRNN.eval()

    def predictByGreedySearch(self, inputSeq, maxAnswerLength=32, showAttention=False, figsize=(12,6)):
        inputSeq = filter_sent(inputSeq)
        inputSeq = [w for w in jieba.lcut(inputSeq) if w in self.word2id.keys()]

        X = seq2id(self.word2id, inputSeq)
        XLens = torch.tensor([len(X)+1], dtype=torch.int, device=self.device)
        X = X + [eosToken]
        X = torch.tensor([X], dtype=torch.long, device=self.device)

        d = int(self.encoderRNN.bidirectional)+1
        hidden = torch.zeros((d*self.encoderRNN.numLayers, 1, self.hiddenSize), dtype=torch.float32, device=self.device)
        encoderOutput, hidden = self.encoderRNN(X, XLens, hidden)
        hidden = hidden[-d*self.decoderRNN.numLayers::2].contiguous()

        attentionArrs = []
        Y = []
        decoderInput = torch.tensor([[sosToken]], dtype=torch.long, device=self.device)
        while decoderInput.item() != eosToken and len(Y)<maxAnswerLength:
            decoderOutput, hidden, decoderAttentionWeight = self.decoderRNN(decoderInput, hidden, encoderOutput)   
            topv, topi = decoderOutput.topk(1)
            decoderInput = topi[:,:,0]
            attentionArrs.append(decoderAttentionWeight.data.cpu().numpy().reshape(1,XLens))
            Y.append(decoderInput.item())
        outputSeq = id2seq(self.id2word, Y)
        if showAttention:
            attentionArrs = np.vstack(attentionArrs)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot('111')
            cax = ax.matshow(attentionArrs, cmap='bone')
            fig.colorbar(cax)
            ax.set_xticklabels(['', '<SOS>'] + inputSeq)
            ax.set_yticklabels([''] + outputSeq)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            plt.show()
        return ''.join(outputSeq[:-1])

    def predictByBeamSearch(self, inputSeq, beamWidth=10, maxAnswerLength=32, alpha=0.7, isRandomChoose=False, allRandomChoose=False, improve=True, showInfo=False):
        outputSize = len(self.id2word)
        inputSeq = filter_sent(inputSeq)
        inputSeq = [w for w in jieba.lcut(inputSeq) if w in self.word2id.keys()]

        X = seq2id(self.word2id, inputSeq)
        XLens = torch.tensor([len(X)+1], dtype=torch.int, device=self.device)
        X = X + [eosToken]
        X = torch.tensor([X], dtype=torch.long, device=self.device)

        d = int(self.encoderRNN.bidirectional)+1
        hidden = torch.zeros((d*self.encoderRNN.numLayers, 1, self.hiddenSize), dtype=torch.float32, device=self.device)
        encoderOutput, hidden = self.encoderRNN(X, XLens, hidden)
        hidden = hidden[-d*self.decoderRNN.numLayers::2].contiguous()

        Y = np.ones([beamWidth, maxAnswerLength], dtype='int32')*eosToken
        # prob: beamWidth × 1
        prob = np.zeros([beamWidth, 1], dtype='float32')
        decoderInput = torch.tensor([[sosToken]], dtype=torch.long, device=self.device)
        # decoderOutput: 1 × 1 × outputSize; hidden: numLayers × 1 × hiddenSize 
        decoderOutput, hidden, decoderAttentionWeight = self.decoderRNN(decoderInput, hidden, encoderOutput)
        # topv: 1 × 1 × beamWidth; topi: 1 × 1 × beamWidth
        topv, topi = decoderOutput.topk(beamWidth)
        # decoderInput: beamWidth × 1
        decoderInput = topi.view(beamWidth, 1)
        for i in range(beamWidth):
            Y[i, 0] = decoderInput[i].item()
        Y_ = Y.copy()
        prob += topv.view(beamWidth, 1).data.cpu().numpy()
        prob_ = prob.copy()
        # hidden: numLayers × beamWidth × hiddenSize
        hidden = hidden.expand(-1, beamWidth, -1).contiguous()
        localRestId = np.array([i for i in range(beamWidth)], dtype='int32')
        encoderOutput = encoderOutput.expand(beamWidth, -1, -1) # => beamWidth × 1 × hiddenSize
        for i in range(1, maxAnswerLength):
            # decoderOutput: beamWidth × 1 × outputSize; hidden: numLayers × beamWidth × hiddenSize; decoderAttentionWeight: beamWidth × 1 × XSeqLen
            decoderOutput, hidden, decoderAttentionWeight = self.decoderRNN(decoderInput, hidden, encoderOutput)
            # topv: beamWidth × 1; topi: beamWidth × 1 
            if improve:
                decoderOutput = decoderOutput.view(-1, 1)
                if allRandomChoose:
                    topv, topi = self._random_pick_k_by_prob(decoderOutput, k=beamWidth)
                else:
                    topv, topi = decoderOutput.topk(beamWidth, dim=0)
            else:
                topv, topi = (torch.tensor(prob[localRestId], dtype=torch.float32, device=self.device).unsqueeze(2)+decoderOutput).view(-1,1).topk(beamWidth, dim=0)
            # decoderInput: beamWidth × 1
            decoderInput = topi%outputSize

            idFrom = topi.cpu().view(-1).numpy()//outputSize
            Y[localRestId, :i+1] = np.hstack([Y[localRestId[idFrom], :i], decoderInput.cpu().numpy()])
            prob[localRestId] = prob[localRestId[idFrom]] + topv.data.cpu().numpy()
            hidden = hidden[:, idFrom, :]

            restId = (decoderInput!=eosToken).cpu().view(-1)
            localRestId = localRestId[restId.numpy().astype('bool')]
            decoderInput = decoderInput[restId]
            hidden = hidden[:, restId, :]
            encoderOutput = encoderOutput[restId]
            beamWidth = len(localRestId)
            if beamWidth<1:
                break
        lens = [i.index(eosToken) if eosToken in i else maxAnswerLength for i in Y.tolist()]
        ans = [''.join(id2seq(self.id2word, i[:l])) for i,l in zip(Y,lens)]
        prob = [prob[i,0]/np.power(lens[i], alpha) for i in range(len(ans))]
        if isRandomChoose or allRandomChoose:
            prob = [np.exp(p) for p in prob]
            prob = [p/sum(prob) for p in prob]
            if showInfo:
                for i in range(len(ans)):
                    print((ans[i], prob[i]))
            return random_pick(ans, prob)
        else:
            ansAndProb = list(zip(ans,prob))
            ansAndProb.sort(key=lambda x: x[1], reverse=True)
            if showInfo:
                for i in ansAndProb:
                    print(i)
            return ansAndProb[0][0]

    def evaluate(self, dataClass, batchSize=128, isDataEnhance=False, dataEnhanceRatio=0.2, streamType='train'):
        dataClass.reset_word_id_map(self.id2word, self.word2id)
        dataStream = dataClass.one_epoch_data_stream(batchSize=batchSize, isDataEnhance=isDataEnhance, dataEnhanceRatio=dataEnhanceRatio, type=streamType)
        bleuScore, embAveScore = 0.0, 0.0
        totalSamplesNum = dataClass.trainSampleNum if streamType=='train' else dataClass.testSampleNum
        iters = 0
        st = time.time()
        while True:
            try:
                X, XLens, Y, YLens = next(dataStream)
            except:
                break
            bleuScore += _bleu_score(self.encoderRNN, self.decoderRNN, X, XLens, Y, YLens, dataClass.maxSentLen, self.device, mean=False)
            embAveScore += _embAve_score(self.encoderRNN, self.decoderRNN, X, XLens, Y, YLens, dataClass.maxSentLen, self.device, mean=False)
            iters += len(X)
            finishedRatio = iters/totalSamplesNum
            print('Finished %.3lf%%; remaining time: %.3lfs'%(finishedRatio*100.0, (time.time()-st)*(1.0-finishedRatio)/finishedRatio))
        return bleuScore/totalSamplesNum, embAveScore/totalSamplesNum
    def _random_pick_k_by_prob(self, decoderOutput, k):
        # decoderOutput: beamWidth*outputSize × 1
        df = pd.DataFrame([[i] for i in range(len(decoderOutput))])
        prob = torch.softmax(decoderOutput.data, dim=0).cpu().numpy().reshape(-1)
        topi = torch.tensor(np.array(df.sample(n=k, weights=prob)), dtype=torch.long, device=self.device)
        return decoderOutput[topi.view(-1)], topi

def random_pick(sample, prob):
    x = random.uniform(0,1)
    cntProb = 0.0
    for sampleItem, probItem in zip(sample, prob):
        cntProb += probItem
        if x < cntProb:break
    return sampleItem

def _bleu_score(encoderRNN, decoderRNN, X, XLens, Y, YLens, maxSentLen, device, mean=True):
    Y_pre = _calculate_Y_pre(encoderRNN, decoderRNN, X, XLens, Y, maxSentLen, teacherForcingRatio=0, device=device)
    Y = [list(Y[i])[:YLens[i]-1] for i in range(len(YLens))]
    Y_pre = Y_pre.cpu().data.numpy()
    Y_preLens = [list(i).index(0) if 0 in i else len(i) for i in Y_pre]
    Y_pre = [list(Y_pre[i])[:Y_preLens[i]] for i in range(len(Y_preLens))]
    bleuScore = [sentence_bleu([i], j, weights=(1,0,0,0)) for i,j in zip(Y, Y_pre)]
    return np.mean(bleuScore) if mean else np.sum(bleuScore)

def _embAve_score(encoderRNN, decoderRNN, X, XLens, Y, YLens, maxSentLen, device, mean=True):
    Y_pre = _calculate_Y_pre(encoderRNN, decoderRNN, X, XLens, Y, maxSentLen, teacherForcingRatio=0, device=device)
    Y_pre = Y_pre.data
    Y_preLens = [list(i).index(0) if 0 in i else len(i) for i in Y_pre]

    emb = encoderRNN.embedding
    Y, Y_pre = emb(torch.tensor(Y, dtype=torch.long, device=device)).cpu().data.numpy(), emb(Y_pre).cpu().data.numpy()

    sentVec = np.array([np.mean(Y[i,:YLens[i]], axis=0) for i in range(len(Y))], dtype='float32')
    sent_preVec = np.array([np.mean(Y_pre[i,:Y_preLens[i]], axis=0) for i in range(len(Y_pre))], dtype='float32')

    embAveScore = np.sum(sentVec*sent_preVec, axis=1)/(np.sqrt(np.sum(np.square(sentVec), axis=1))*np.sqrt(np.sum(np.square(sent_preVec), axis=1)))
    return np.mean(embAveScore) if mean else np.sum(embAveScore)
    
def _calculate_loss(encoderRNN, decoderRNN, X, XLens, Y, YLens, teacherForcingRatio, device):
    featureSize, hiddenSize = encoderRNN.featureSize, encoderRNN.hiddenSize
    # X: batchSize × XSeqLen; Y: batchSize × YSeqLen
    X, Y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(Y, dtype=torch.long, device=device)
    XLens, YLens = torch.tensor(XLens, dtype=torch.int, device=device), torch.tensor(YLens, dtype=torch.int, device=device)

    batchSize = X.size(0)
    XSeqLen, YSeqLen = X.size(1), YLens.max().item()
    encoderOutput = torch.zeros((batchSize, XSeqLen, featureSize), dtype=torch.float32, device=device)

    d = int(encoderRNN.bidirectional)+1
    hidden = torch.zeros((d*encoderRNN.numLayers, batchSize, hiddenSize), dtype=torch.float32, device=device)

    XLens, indices = torch.sort(XLens, descending=True)
    _, desortedIndices = torch.sort(indices, descending=False)
    encoderOutput, hidden = encoderRNN(X[indices], XLens, hidden)
    encoderOutput, hidden = encoderOutput[desortedIndices], hidden[-d*decoderRNN.numLayers::d, desortedIndices, :] #hidden[:decoderRNN.numLayers, desortedIndices, :]
    decoderInput = torch.tensor([[sosToken] for i in range(batchSize)], dtype=torch.long, device=device)
    loss, nTotal = 0, 0
    for i in range(YSeqLen):
        # decoderOutput: batchSize × 1 × outputSize
        decoderOutput, hidden, decoderAttentionWeight = decoderRNN(decoderInput, hidden, encoderOutput)
        loss += F.nll_loss(decoderOutput[:,0,:], Y[:,i], reduction='sum')
        nTotal += len(decoderInput)
        if random.random() < teacherForcingRatio:
            decoderInput = Y[:,i:i+1]
        else:
            topv, topi = decoderOutput.topk(1)
            decoderInput = topi[:,:,0]# topi.squeeze().detach()
        restId = (YLens>i+1).view(-1)
        decoderInput = decoderInput[restId]
        hidden = hidden[:, restId, :]
        encoderOutput = encoderOutput[restId]
        Y = Y[restId]
        YLens = YLens[restId]
    return loss, nTotal

def _calculate_Y_pre(encoderRNN, decoderRNN, X, XLens, Y, YMaxLen, teacherForcingRatio, device):
    featureSize, hiddenSize = encoderRNN.featureSize, encoderRNN.hiddenSize
    # X: batchSize × XSeqLen; Y: batchSize × YSeqLen
    X, Y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(Y, dtype=torch.long, device=device)
    XLens = torch.tensor(XLens, dtype=torch.int, device=device)

    batchSize = X.size(0)
    XSeqLen = X.size(1)
    encoderOutput = torch.zeros((batchSize, XSeqLen, featureSize), dtype=torch.float32, device=device)

    d = int(encoderRNN.bidirectional)+1
    hidden = torch.zeros((d*encoderRNN.numLayers, batchSize, hiddenSize), dtype=torch.float32, device=device)

    XLens, indices = torch.sort(XLens, descending=True)
    _, desortedIndices = torch.sort(indices, descending=False)
    encoderOutput, hidden = encoderRNN(X[indices], XLens, hidden)
    encoderOutput, hidden = encoderOutput[desortedIndices], hidden[-d*decoderRNN.numLayers::d, desortedIndices, :] #hidden[:decoderRNN.numLayers, desortedIndices, :]
    decoderInput = torch.tensor([[sosToken] for i in range(batchSize)], dtype=torch.long, device=device)
    Y_pre, localRestId = torch.ones([batchSize, YMaxLen], dtype=torch.long, device=device)*eosToken, torch.tensor([i for i in range(batchSize)], dtype=torch.long, device=device)
    for i in range(YMaxLen):
        # decoderOutput: batchSize × 1 × outputSize
        decoderOutput, hidden, decoderAttentionWeight = decoderRNN(decoderInput, hidden, encoderOutput)
        if random.random() < teacherForcingRatio:
            decoderInput = Y[:,i:i+1]
        else:
            topv, topi = decoderOutput.topk(1)
            decoderInput = topi[:,:,0]# topi.squeeze().detach()
        Y_pre[localRestId, i] = decoderInput.squeeze()
        restId = (decoderInput!=eosToken).view(-1)
        localRestId = localRestId[restId]
        decoderInput = decoderInput[restId]
        hidden = hidden[:, restId, :]
        encoderOutput = encoderOutput[restId]
        Y = Y[restId]
        if len(localRestId)<1:
            break
    return Y_pre