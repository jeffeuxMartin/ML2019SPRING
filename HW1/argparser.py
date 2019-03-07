from __future__ import print_function
from argparse import ArgumentParser

# hw1.py test predict -t train -lr lr -lb lambda -it iter -w weight -m mode
parser = ArgumentParser()
parser.add_argument("test", help="testing file")
parser.add_argument("test", help="testing file")
parser.add_argument("-t", "--train", help="training file", dest="train", default="train.csv")
parser.add_argument("-m", "--mode", help="mode if training or testing", dest="mode", default="test")
parser.add_argument("-it", help="iteration of training", dest="it", default="50000")
parser.add_argument("-lr", "--lr", help="learning rate", dest="lr", default="300")
parser.add_argument("-lb", "--lambda", help="lambda of regularization", dest="lambda", default="0.1")
parser.add_argument("-w", "--weight", help="weight file", dest="weight", default="weight.npy")

args = parser.parse_args()
