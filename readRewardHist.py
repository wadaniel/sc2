import argparse
import numpy as np

####### Parsing arguments

parser = argparse.ArgumentParser()
parser.add_argument('-fn', '--filename', help='File to read.', required=True, type=str)
parser.add_argument('-tf', '--testingFrequency', help='Testing Frequency as configured in korali app.', required=True, type=int)
parser.add_argument('-epg', '--episodesPerGeneration', help='Episodes Per Generation as configured in korali app.', required=True, type=int)
parser.add_argument('-pte', '--policyTestingEpisodes', help='Episodes Per Generation as configured in korali app.', required=True, type=int)

args = parser.parse_args()
print(args)

winThreshold = 19.99

testingRollouts = args.policyTestingEpisodes * args.episodesPerGeneration
testStat = np.load(args.filename)
rewardHistory = testStat['rewardHistory']

print("lenWinHistory {}".format(len(rewardHistory)))
print("testingRollOuts {}".format(testingRollouts))

assert(len(rewardHistory) % testingRollouts == 0)

rewardHistory = np.reshape(rewardHistory, (-1,testingRollouts))
winHistory = rewardHistory > winThreshold
print(rewardHistory)
print(winHistory)
#winRate = np.mean(rewardHistory, axis=1)
#winSdev = np.std(rewardHistory, axis=1)
#print(sampleHistory)
#print(winHistory)
#print(winRate)
 




