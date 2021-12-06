#!/usr/bin/env python3

from smac.env import StarCraft2Env
import os.path
import numpy as np
import math

### Environment specific korali configuratoin
def initEnvironment(e, mapName, multPolicies):
 
  env = StarCraft2Env(map_name=mapName)
  env_info = env.get_env_info()
  print(env_info)

  actionVariableCount = 1
  numPossibleActions = env_info["n_actions"]
  possibleActions = [ [idx] for idx in range(numPossibleActions) ]
  numIndividuals = env_info["n_agents"]
  stateVariableCount = env_info["state_shape"]
  observationVariableCount = env_info["obs_shape"]

  print(numPossibleActions)
  print(possibleActions)
  print(numIndividuals)
  print(stateVariableCount)
  print(observationVariableCount)

  # Defining problem configuration for pettingZoo environments
  e["Problem"]["Environment Function"] = lambda s : environment(s, env)
  e["Problem"]["Possible Actions"] = possibleActions
  e["Problem"]["Agents Per Environment"] = numIndividuals
  if (multPolicies == 1) :
      e["Problem"]["Policies Per Environment"] = numIndividuals
 
  # Defining State Variables
 
  for i in range(observationVariableCount):
     e["Variables"][i]["Name"] = "State Variable " + str(i)
     e["Variables"][i]["Type"] = "State"
  
  # Defining Action Variables
 
  for i in range(actionVariableCount):
    e["Variables"][observationVariableCount + i]["Name"] = "Action Variable " + str(i)
    e["Variables"][observationVariableCount + i]["Type"] = "Action"

  # Adjust Bacth Size  
  # e["Solver"]["Mini Batch"]["Size"] = int(256 / numIndividuals)

# Helper function to extract available actions
def getAvailableActions(nAgents, env):
    availableActions = []
    for agentId in range(nAgents):
        availableActions.append(env.get_avail_agent_actions(agentId))

    return availableActions

### The Environment
def environment(s, env):
        
    env.reset()
    env_info = env.get_env_info()
    nAgents = env_info["n_agents"]
    
    terminated = False
    episodeReward = 0
 
    obs = env.get_obs()
    obs = [ o.tolist() for o in obs ]

    s["State"] = obs
    s["Available Actions"] = getAvailableActions(nAgents, env)
    
    step = 0
    while not terminated:
        
        s.update()

        actions = s["Action"]
        actions = [a[0] for a in actions]
        reward, terminated, _ = env.step(actions)
        episodeReward += reward
 
        obs = env.get_obs()
        obs = [ o.tolist() for o in obs ]
        s["Reward"] = [ reward ] * nAgents
 
        s["State"] = obs
        s["Available Actions"] = getAvailableActions(nAgents, env)
        step = step + 1
 
    
    s["Termination"] = "Terminal"
    
    if s["Mode"] == "Testing":
        print("Testing episode reward after {} steps: {}".format(step, episodeReward))
        sampleId = s["Sample Id"]
        resDir = s["Custom Settings"]["Result Folder"]
        testingHistFile = resDir + 'testingRewardHistory.npz'

        history = None
        sampleHistory = None
        if os.path.isfile(testingHistFile):
            # Append results
            testingStat = np.load(testingHistFile)
           
            sampleHistory = testingStat['sampleHistory']
            sampleHistory = np.append(sampleHistory, sampleId)

            rewardHistory = testingStat['rewardHistory']
            rewardHistory = np.append(rewardHistory, episodeReward)
 
        else:
            # Init
            sampleHistory = [ sampleId ]
            rewardHistory = [ episodeReward ]

        np.savez(testingHistFile, sampleHistory = sampleHistory, rewardHistory = rewardHistory)
