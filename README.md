
Attention-Seq2Seq-Chatbot-by-Pytorch1.0.1
===
![totalModel](https://github.com/wudejian789/Attention-Seq2Seq-Chatbot-by-Pytorch1.0.1/blob/master/image/totalModel.png)
>Note: If there's any error like "No module named 'xxx'", please use command "pip install xxx" to repair.  

Feel free to add my QQ: ***793729558*** to discuss with me.  
Also you can add the QQ group: ***647303915*** to discuss together.  
My paper is [here](https://github.com/wudejian789/Attention-Seq2Seq-Chatbot-by-Pytorch1.0.1/blob/master/基于深度学习的聊天机器人实现\(第三稿\).pdf).  

# 1. For Entertainment
## 1.1 Download the trained model
My trained model is saved in my Baidu Net Disk.  
The model below is trained in qingyun corpus.  

|encoder|decoder|attention|data enhance|test size|address|key|  
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|  
|5×Bi_GRU|3×GRU|Bahdanau(concat)|False|0.1|[link](https://pan.baidu.com/s/1qel4uPNAdVF7Sjl-fzWAuQ)|s55l|  
|5×Bi_GRU|3×GRU|Luong(dot)|False|0.1|[link](https://pan.baidu.com/s/1ftVs682QzmFDqPRdSgN7Zg)|x76r|  
|5×Bi_GRU|3×GRU|Luong(general)|False|0.1|[link](https://pan.baidu.com/s/1uVg4IwnPzCx7H48wFmjWOA)|p3y0|  
|5×Bi_GRU|3×GRU|Luong(concat)|False|0.1|[link](https://pan.baidu.com/s/16SnTTx8CQBhnkEOe6Dj0QA)|xte1|  
|5×Bi_GRU|3×GRU|Luong(general)|False|0.0|[link](https://pan.baidu.com/s/1pn4_6JCco95g9JHxC0R9FQ)|pl5j|  
|5×Bi_GRU|3×GRU|Luong(general)|True|0.0|[link](https://pan.baidu.com/s/1_GHEDRzQyl-R5LIndgQurQ)|0sfe|  

## 1.2 Running the demo program
I have offered 2 demo programs: "demoCmd.py" and "demoWeChat.py".  
Use command " ***python demoCmd.py --model="xxx.pkl" --device="cpu"(or "cuda")*** " to run the cmd demo program.
>***model*** is your model path;  
>***device*** is your program running environment, "cpu" for CPU or "cuda" for GPU;  

![run shot](https://github.com/wudejian789/Attention-Seq2Seq-Chatbot-by-Pytorch1.0.1/blob/master/image/demoCmd.png)
Also you can run "demoWeChat.py" to use the chatbot to reply your WeChat message. The parameters is the same as above.  
# 2. For Research
## 2.1 Import the module
```python
from model.nnModel import *
from model.corpusSolver import *
import torch
```
## 2.2 How to load the data
```python
dataClass = Corpus('./corpus/qingyun.tsv', maxSentenceWordsNum=25)
```
>***First parameter*** is your corpus path;  
>***maxSentenceWordsNum*** will ignore the data whose words number of question or answer is too big;  

Also you can load your corpus. Only your file content formats need to be consistent:
>Q  A  
>Q  A  
>...

Every line is a question and a answer with a '\t' split.  
The corpus comes from [https://github.com/codemayq/chinese_chatbot_corpus](https://github.com/codemayq/chinese_chatbot_corpus), you can get larger corpus from this Github.  
Thanks very much for the corpus summarized and processed by the author.  
## 2.3 How to train your model
First you need to create a Seq2Seq object.
```python
model = Seq2Seq(dataClass, featureSize=256, hiddenSize=256, 
                attnType='L', attnMethod='general', 
                encoderNumLayers=3, decoderNumLayers=2, 
                encoderBidirectional=True, 
                device=torch.device('cuda:0'))
```
>***First parameter*** is your corpus class object.  
>***featureSize*** is your word vector size;  
>***hiddenSize*** is your RNN hidden state size;  
>***attnType*** is your attention type. It can be 'B' for using Bahdanau Attention Structure or 'L' for using Luong Structure;  
> ***attnMethod*** is Luong Attention Method. It can be 'dot', 'general' or 'concat'.  
>***encoderNumLayers*** is the layer number of your encoder RNN;  
>***decoderNumlayers*** is the layer number of your decoder RNN;  
>***encoderBidirectional*** is if your encoder RNN is bidirectional;  
>***device*** is your building environment. If using CPU, then device=torch.device('cpu'); if using GPU, then device=torch.device('cuda:0');  

Then you can train your model.
```python
model.train(batchSize=1024, epoch=500)
```
>***batchSize*** is the number of data used for each train step;  
>***epoch*** is the total iteration number of your training data;  

And the log will be print like follows:
```
...
After iters 6540: loss = 0.844; train bleu: 0.746, embAve: 0.754; 2034.582 qa/s; remaining time: 48096.110s;
After iters 6550: loss = 0.951; train bleu: 0.734, embAve: 0.755; 2034.518 qa/s; remaining time: 48092.589s;
After iters 6560: loss = 1.394; train bleu: 0.735, embAve: 0.759; 2034.494 qa/s; remaining time: 48088.128s;
...
```
Finally you need to save your model for future use.
```
model.save('model.pkl')
```
>First parameter is the name of model saved.  

Ok, I know you are too lazy to train your own model. Also you can download my trained model in section **1.1**.

## 2.4 How to use your model to build a chatbot
First you need to create a Chatbot object.
```python
chatbot = Chatbot('model.pkl')
```
>First parameter is your model path;  

Then you can use the greedy search to generate the answer.
```python
chatbot.predictByGreedySearch("你好啊")
```
>First parameter is your question;  

It will return the answer like "你好,我就开心了". Also you can plot the attention by ***showAttention=True***.
Or you can use the beam search to generate the answer.
```python
chatbot.predictByBeamSearch("什么是ai", isRandomChoose=True, beamWidth=10)
```
>***First parameter*** is your question;  
>***isRandomChoose*** determines whether probability sampling is performed in the final beamwidth answers.  
>***beamWidth*** is the search width in beam search;   

It will return the answer like "反正不是苹果". Also you can show the probabilities of the beamwidth answers by ***showInfo=True***.
## 2.5 How to use a trained word embedding
First you need to calculate 4 variables: 
>***id2word***: a list of word, and the first two words have to be "\<SOS\>" and "\<EOS\>", e.g., ["\<SOS\>", "\<EOS\>", "你", "天空", "人工智能", "中国", ...];  
>***word2id***: a dict with the key of ***word*** and the value of ***id***, corresponding to ***id2word***, e.g., {"\<SOS\>":0, "\<EOS\>":1, "你":2, "天空":3, "人工智能":4, "中国":5, ...};  
>***wordNum***: the total number of words. It is equal to len(id2word) or len(word2id);  
>***wordEmb***: the word embedding array with shape (wordNum, featureSize) and the word order need to be consistent with id2word or word2id;  you can random initialize the vector or "\<SOS\>" and "\<EOS\>";  

Then add first three variables as parameters when you load the data. 
```python
dataClass = Corpus(..., id2word=id2word, word2id=word2id, wordNum=wordNum)
```
Next you need to create the word embedding object.
```python
embedding = torch.nn.Embedding.from_pretrained(torch.tensor(wordEmb))
```
Finally add the embedding parameter when you create the Seq2Seq object.
```python
model = Seq2Seq(..., embedding=embedding)
```
Also you can download a trained word embedding from [https://github.com/Embedding/Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors).  
Thanks very much for the trained word embedding provided by the author.  
## 2.6 Other function
For other functions such as data enhance, etc, please dig for yourselves.
# 3. Reference
[1] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate[J]. arXiv preprint arXiv:1409.0473, 2014.  
[2] Luong M T, Pham H, Manning C D. Effective approaches to attention-based neural machine translation[J]. arXiv preprint arXiv:1508.04025, 2015.  
[3] [https://pytorch.apachecn.org/docs/1.0/#/](https://pytorch.apachecn.org/docs/1.0/#/)  
