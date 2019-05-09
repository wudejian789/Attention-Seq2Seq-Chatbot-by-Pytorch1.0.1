from model.nnModel import ChatBot
import torch,warnings,argparse
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='The path of your model file.', required=True, type=str)
parser.add_argument('--device', help='Your program running environment, "cpu" or "cuda"', type=str, default='cpu')
args = parser.parse_args()
print(args)

if __name__ == "__main__":
    print('Loading the model...')
    chatBot = ChatBot(args.model, device=torch.device(args.device))
    print('Finished...')

    allRandomChoose,showInfo = False, False
    while True:
        inputSeq = input("You: ")
        if inputSeq=='_crazy_on_':
            allRandomChoose = True
            print('AI: ','成功开启疯狂模式...')
        elif inputSeq=='_crazy_off_':
            allRandomChoose = False
            print('AI: ','成功关闭疯狂模式...')
        elif inputSeq=='_showInfo_on_':
            showInfo = True
            print('AI: ','成功开启日志打印...')
        elif inputSeq=='_showInfo_off_':
            showInfo = False
            print('AI: ','成功关闭日志打印...')
        else:
            outputSeq = chatBot.predictByBeamSearch(inputSeq, isRandomChoose=True, allRandomChoose=allRandomChoose, showInfo=showInfo)
            print('AI: ',outputSeq)
        print()