export TRAINING_DATA=/Users/sushantverma/PycharmProjects/abhishek_thakur/vikrant/ml/input/train_folds.csv
export TEST_DATA=/Users/sushantverma/PycharmProjects/abhishek_thakur/vikrant/ml/input/test.csv
export MODEL=$1



#FOLD=0 python -m src.train
#FOLD=1 python -m src.train
#FOLD=2 python -m src.train
#FOLD=3 python -m src.train
#FOLD=4 python -m src.train
python -m src.predict
