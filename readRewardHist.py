import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
####### Parsing arguments

parser = argparse.ArgumentParser()
parser.add_argument('-fn', '--filename', help='File to read.', required=True, type=str)
parser.add_argument('-pte', '--policyTestingEpisodes', help='Episodes Per Generation as configured in korali app.', required=True, type=int)

args = parser.parse_args()

winThreshold = 19.99

testingRollouts = args.policyTestingEpisodes
testStat = np.load(args.filename)
rewardHistory = testStat['rewardHistory']

print("lenWinHistory {}".format(len(rewardHistory)))
print("testingRollOuts {}".format(testingRollouts))

assert(len(rewardHistory) % testingRollouts == 0)

rewardHistory = np.reshape(rewardHistory, (-1,testingRollouts))
winHistory = rewardHistory > winThreshold

winProb = np.mean(winHistory, axis=1)

print(len(winProb))
print(winProb)

fig, ax = plt.subplots()
ax.plot(winProb, linewidth=2.0)
ax.set_ylim(0., 1.)
fig.savefig("winProbabilities.png")
