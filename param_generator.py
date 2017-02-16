wordCNNSize = [100]
uttCNNSize = [100]
dropout = [0.3, 0.6, 0.9]
uttWindowSize = [10]
learningRate = [0.0001, 0.0003, 0.0009]

p = 3
cnt = 0

for wordCNN in wordCNNSize:
    for uttCNN in uttCNNSize:
        for drop in dropout:
            for uttWin in uttWindowSize:
                for lr in learningRate:
                    if cnt % p == 0:
                        print('(nohup python3 main.py --device /gpu:1 --wordCNNSize {} --uttCNNSize {} --dropOut {} --uttWindowSize {} --learningRate {} &'
                              .format(wordCNN, uttCNN, drop, uttWin, lr))
                    elif cnt %p == p-1:
                        print('nohup python3 main.py --device /gpu:1 --wordCNNSize {} --uttCNNSize {} --dropOut {} --uttWindowSize {} --learningRate {} &);\n'
                              .format(wordCNN, uttCNN, drop, uttWin, lr))
                    else:
                        print('nohup python3 main.py --device /gpu:1 --wordCNNSize {} --uttCNNSize {} --dropOut {} --uttWindowSize {} --learningRate {} &'
                              .format(wordCNN, uttCNN, drop, uttWin, lr))
                    cnt += 1
