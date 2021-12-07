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

testingRollouts = args.policyTestingEpisodes
winStat = np.load(args.filename)
sampleHistory = winStat['sampleHistory']
print(sampleHistory)
winHistory = winStat['history']
print(winHistory)
print("lenWinHistory {}".format(len(winHistory)))
print("testingRollOuts {}".format(testingRollouts))
assert(len(winHistory) % testingRollouts == 0)

winHistory = np.reshape(winHistory, (-1,testingRollouts))
winRate = np.mean(winHistory, axis=1)
winSdev = np.std(winHistory, axis=1)
print(sampleHistory)
print(winHistory)
print(winRate)
 




