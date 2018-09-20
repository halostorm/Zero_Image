

trainB_path = r'../DatasetB_20180919/train.txt'

trainA_path = r'../DatasetA_train_20180813/train.txt'

with open(trainB_path,'a+') as FB:
    with open(trainA_path,'r')as FA:
        for line in FA:
            FB.write(line)