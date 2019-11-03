# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import Actions
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

import sys

sys.path.append("teams/<puffrora>/")

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'Attacker', second = 'Defender'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)

#########################
# Method 1 - Side Agent #
#########################

class SideAgent(CaptureAgent):

  def registerInitialState(self, gameState):

    CaptureAgent.registerInitialState(self, gameState)

    self.searchTree = SearchTree()

    self.allyIndex = (self.index+2)%4
    if self.index < 2:
      self.side = "Top"
    else:
      self.side = "Bottom"

    self.foodTotal = len(self.getFood(gameState).asList())
    self.foodDefendingPrev = self.getFoodYouAreDefending(gameState).asList()
    self.foodDefendingNow = self.getFoodYouAreDefending(gameState).asList()
    self.foodLost = None
    self.patience = 0

  def chooseAction(self, gameState):
    start = time.time()

    self.foodDefendingNow = self.getFoodYouAreDefending(gameState).asList()
    foodLost, patience = self.closestFoodLost(gameState)

    if foodLost!=None:
      self.foodLost = foodLost
      self.patience = patience

    if self.patience > 0:
      self.patience -= 1
    else:
      self.patience = 0
      self.foodLost = None

    # print self.foodLost, self.patience
    myState = gameState.getAgentState(self.index)
    if myState.isPacman:
      best_action = self.actionAsPacman(gameState)
    else:
      best_action = self.actionAsGhost(gameState)
    self.foodDefendingPrev = self.foodDefendingNow
    print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    return best_action

  """
  ==============
  Help functions
  ==============
  """
  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def topFoods(self, gameState):
    foodList = self.getFood(gameState).asList()
    topFoods = [(x,y) for (x,y) in foodList if y>=gameState.data.layout.height/2]
    return topFoods

  def bottomFoods(self, gameState):
    foodList = self.getFood(gameState).asList()
    bottomFoods = [(x,y) for (x,y) in foodList if y<gameState.data.layout.height/2]
    return bottomFoods

  def sideFoods(self, gameState):
    foodList = []
    if self.side == "Top":
      foodList = self.topFoods(gameState)
    else:
      foodList = self.bottomFoods(gameState)
    return foodList

  def getBorders(self, gameState):
    width = gameState.data.layout.width
    height = gameState.data.layout.height
    if self.red:
      return [(width/2-1, h) for h in range(0, height) if (width/2-1, h) not in gameState.data.layout.walls.asList()]
    else:
      return [(width/2, h) for h in range(0, height) if (width/2, h) not in gameState.data.layout.walls.asList()]

  def getCapsules(self, gameState):
    if self.red:
      return gameState.getBlueCapsules()
    else:
      return gameState.getRedCapsules()

  def closestFoodLost(self, gameState):
    myPos = gameState.getAgentState(self.index).getPosition()
    allyPos = gameState.getAgentState(self.allyIndex).getPosition()
    foodsLost = [food for food in self.foodDefendingPrev if food not in self.foodDefendingNow]

    foodsLostNearToMe = []
    for foodLost in foodsLost:
      myDistance = self.getMazeDistance(myPos, foodLost)
      allyDistance = self.getMazeDistance(allyPos, foodLost)
      if myDistance <= allyDistance:
        foodsLostNearToMe.append(foodLost)

    if len(foodsLostNearToMe)!=0:
      patience, foodLostClosestToMe = min((self.getMazeDistance(myPos, foodLost), foodLost) for foodLost in foodsLostNearToMe)
    else:
      patience = 0
      foodLostClosestToMe = None

    return foodLostClosestToMe, patience - 3

  def invadersNearby(self, gameState):
    myPos = gameState.getAgentState(self.index).getPosition()
    allyPos = gameState.getAgentState(self.allyIndex).getPosition()
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invadersPos = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None and a.scaredTimer==0]

    nearInvaders = []
    for invader in invadersPos:
      myDistance = self.getMazeDistance(myPos, invader)
      allyDistance = self.getMazeDistance(allyPos, invader)
      if myDistance <= allyDistance:
        nearInvaders.append(invader)
      # elif (myPos[0]-invader[0])*(allyPos[0]-invader[0])<=0 or (myPos[1]-invader[1])*(allyPos[1]-invader[1])<=0 :
      #   nearInvaders.append(invader)
    return nearInvaders

  def protectorsNearby(self, gameState, isPacman):
    myPos = gameState.getAgentState(self.index).getPosition()
    allyPos = gameState.getAgentState(self.allyIndex).getPosition()
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    protectorsPos = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer==0]

    nearProtectors = []
    for protector in protectorsPos:
      myDistance = self.getMazeDistance(myPos, protector)
      allyDistance = self.getMazeDistance(allyPos, protector)
      if not isPacman and myDistance <= allyDistance :
        nearProtectors.append(protector)
      elif isPacman:
        nearProtectors.append(protector)
    return nearProtectors

  def findClosestFood(self, gameState, isSideFood):
    allyPos = gameState.getAgentState(self.allyIndex).getPosition()

    bestAction = None
    minDistanceToFood = 1200
    for action in gameState.getLegalActions(self.index):
      if action=='Stop':
        continue
      successor = self.getSuccessor(gameState, action)
      myPos = successor.getAgentState(self.index).getPosition()
      distanceToAlly = self.getMazeDistance(myPos, allyPos)

      if isSideFood:
        foodList = self.sideFoods(gameState)
      else:
        foodList = self.getFood(gameState).asList()
      distanceToFood, food  = min([(self.getMazeDistance(myPos, food), food) for food in foodList])
      # capsules = self.getCapsules(gameState)
      # if len(capsules)!=0:
      #   distanceToCapsule, capsule = min([(self.getMazeDistance(myPos, capsule), capsule) for capsule in capsules])

      #   distanceToTarget = distanceToFood if distanceToFood < distanceToCapsule else distanceToCapsule 
      #   target = food if distanceToFood < distanceToCapsule else capsule 
      # else:
      #   distanceToTarget = distanceToFood
      #   target = food

      if minDistanceToFood > distanceToFood:
        minDistanceToFood = distanceToFood
        minFood = food
        bestAction = action

    return bestAction, minDistanceToFood, minFood

  """
  ===============
  Value functions
  ===============
  """
  def directlyBack_valueFunc(self, gameState, node):
    if node.myPos in node.enemiesPos:
      return -10000
      
    disToBorder = min([self.getMazeDistance(node.myPos, border) for border in self.getBorders(gameState)])

    if node.myPos in self.getBorders(gameState):
      return 1000 -disToBorder - 0.5*node.branchDepth

    return -disToBorder - 0.5*node.branchDepth

  def back_valueFunc(self, gameState, node):
    if node.myPos in node.enemiesPos:
      return -10000

    if node.myPos in self.getBorders(gameState):
      return 1000 + (-2)*disToBorder + 10*node.carrying - 0.5*node.branchDepth

    carrying = gameState.getAgentState(self.index).numCarrying
    disToBorder = min([self.getMazeDistance(node.myPos, border) for border in self.getBorders(gameState)])
    return  (-2)*disToBorder + 10*node.carrying - 0.5*node.branchDepth
    # return -disToBorder + 3*node.carrying - 0.5*node.branchDepth

  def capsule_valueFunc(self, gameState, node):
    if node.myPos in node.enemiesPos:
      return -10000

    if len(node.capsules)==0:
      return 10*node.carrying + 20*node.capsuleCarrying - 0.5*node.branchDepth

    disToClosestCapsule = min([self.getMazeDistance(node.myPos, capsule) for capsule in node.capsules])
    return  -disToClosestCapsule + 10*node.carrying + 20*node.capsuleCarrying - 0.5*node.branchDepth

  def food_valueFunc(self, gameState, node):
    if node.myPos in node.enemiesPos:
      return -10000

    if len(node.foodList)!=0:
      disToClosestFood = min([self.getMazeDistance(node.myPos, food) for food in node.foodList])
      return -disToClosestFood + 10*node.carrying - 0.5*node.branchDepth
    else:
      disToCloestBorder = min([self.getMazeDistance(node.myPos, border) for border in self.getBorders(gameState)])
      return -disToClosestFood - 10*node.carrying - 0.5*node.branchDepth

  # def food_valueFunc(self, gameState, node):
  #   if node.myPos in node.enemiesPos:
  #     return -10000

  #   if len(node.foodList)!=0:
  #     disToCloestFood = min([self.getMazeDistance(node.myPos, food) for food in node.foodList])
  #     if len(node.capsules)!=0:
  #       disToClosestCapsule = min([self.getMazeDistance(node.myPos, capsule) for capsule in node.capsules])
  #       disToFoodOrCapsule = -disToCloestFood if disToCloestFood < disToClosestCapsule else -disToClosestCapsule
  #     else:
  #       disToFoodOrCapsule = -disToCloestFood
  #   else:
  #       disToCloestBorder = min([self.getMazeDistance(node.myPos, border) for border in self.getBorders(gameState)])
  #       return -disToCloestBorder
  #   return disToFoodOrCapsule + 100*node.carrying

    
  """
  =========================
  Action decision as Pacman
  =========================
  """
  def actionAsPacman(self, gameState):
    """
    Finds an action to take as a pacman.
    """
    nearProtectors = self.protectorsNearby(gameState, True)
    foodList = self.sideFoods(gameState)
    if len(foodList)==0:
      return self.searchTree.search(gameState, self, nearProtectors, foodList, self.directlyBack_valueFunc)

    timeLeft = gameState.data.timeleft/4+1
    myPos = gameState.getAgentState(self.index).getPosition()
    _, distanceToTarget, target = self.findClosestFood(gameState, True)
    distanceTargetToBorder = min([self.getMazeDistance(target, border) for border in self.getBorders(gameState)])
    if distanceToTarget + distanceTargetToBorder >= timeLeft:
      return self.searchTree.search(gameState, self, nearProtectors, foodList, self.directlyBack_valueFunc)
    # if self.foodLost!=None and gameState.getAgentState(self.index).scaredTimer==0:
    #   return self.searchTree.search(gameState, self, nearProtectors, foodList, self.directlyBack_valueFunc)
    return self.actionAttack(gameState, nearProtectors)

  def actionAttack(self, gameState, nearProtectors):
    myPos = gameState.getAgentState(self.index).getPosition()
    if len(nearProtectors)!=0:
      minDisToProtector = min([self.getMazeDistance(myPos, protector) for protector in nearProtectors])
    else:
      minDisToProtector = 1000
    carrying = gameState.getAgentState(self.index).numCarrying

    # if carrying >= 6:
    #   # print "carrying >= 5"
    #   return self.searchTree.search(gameState, self, nearProtectors, self.getFood(gameState).asList(), self.back_valueFunc)
    # else:
    if len(self.getCapsules(gameState))!=0:
      if minDisToProtector <= 5:
        # print "minDisToProtector <= 5"
        return self.searchTree.search(gameState, self, nearProtectors, self.sideFoods(gameState), self.capsule_valueFunc)
    else:
      if minDisToProtector <= 5 and carrying > 0:
        # print "minDisToProtector <= 2"
        return self.searchTree.search(gameState, self, nearProtectors, self.sideFoods(gameState), self.back_valueFunc)

    bestAction, dis, food = self.findClosestFood(gameState, True)
    # print "findClosest food", food, dis
    return bestAction
    # return self.searchTree.search(gameState, self, nearProtectors, self.getFood(gameState).asList(), self.food_valueFunc)


  """
  ========================
  Action decision as Ghost
  ========================
  """
  def actionAsGhost(self, gameState):
    """
    Finds an action to take as a ghost.
    """
    foodList = self.sideFoods(gameState)
    if len(foodList)!=0 and gameState.getAgentState(self.index).scaredTimer>0:
      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      nearEnemies = [a.getPosition() for a in enemies if a.getPosition()!= None and a.scaredTimer==0]
      return self.actionAttack(gameState, nearEnemies)
      
    nearInvaders = self.invadersNearby(gameState)
    if len(nearInvaders)!=0:
      return self.actionWithInvaders(gameState, nearInvaders)
    if self.foodLost!=None:
      return self.actionWithInvaders(gameState, [self.foodLost])

    nearProtectors = self.protectorsNearby(gameState, False)
    if len(nearProtectors)!=0:
      return self.actionWithProtectors(gameState, nearProtectors)

    if len(foodList)!=0:
      bestAction, _, _ = self.findClosestFood(gameState, True)
    else:
      bestAction = 'Stop'
    return bestAction

  def actionWithInvaders(self, gameState, nearInvaders):
    bestAction = None
    minDistance = 1200
    for action in gameState.getLegalActions(self.index):
      if action == "Stop":
        continue
      successor = self.getSuccessor(gameState, action)
      myState = successor.getAgentState(self.index)
      if myState.isPacman:
        continue
      myPos = myState.getPosition()

      minDistanceToInvaders = min([self.getMazeDistance(myPos, nearInvader) for nearInvader in nearInvaders])
      if minDistanceToInvaders < minDistance:
        minDistance = minDistanceToInvaders
        bestAction = action
      elif minDistanceToInvaders == minDistance and action in ["North", "South"]:
        minDistance = minDistanceToInvaders
        bestAction = action
    return bestAction

  def actionWithProtectors(self, gameState, nearProtectors):
    myPos = gameState.getAgentState(self.index).getPosition()
    if myPos not in self.getBorders(gameState):
      bestAction, _, _ = self.findClosestFood(gameState, True)
      return bestAction
    else:
      return self.searchTree.search(gameState, self, nearProtectors, self.sideFoods(gameState), self.food_valueFunc)

  # def actionWithProtectors(self, gameState, nearProtectors):
  #   bestAction = None
  #   minDistance = 1200

  #   borders = self.getBorders(gameState)
  #   _, nearestBordersToProtectors = min([(self.getMazeDistance(border, protector), border) for border in borders for protector in nearProtectors])

  #   for action in gameState.getLegalActions(self.index):
  #     successor = self.getSuccessor(gameState, action)
  #     myPos = successor.getAgentState(self.index).getPosition()
  #     distanceToBorder = self.getMazeDistance(myPos, nearestBordersToProtectors)
  #     if distanceToBorder < minDistance:
  #       minDistance = distanceToBorder
  #       bestAction = action
  #   return bestAction

class SearchTree:

  def search(self, gameState, agent, root_enemiesPos, root_foodList, evaluate_fun):
    root_myPos = gameState.getAgentState(agent.index).getPosition()
    root_node = TreeNode(root_myPos, root_enemiesPos)
    root_node.foodList = root_foodList
    root_node.capsules = agent.getCapsules(gameState)
    root_node.carrying = gameState.getAgentState(agent.index).numCarrying
    root_node.value = evaluate_fun(gameState, root_node)

    start = time.time()
    q = util.Queue()
    q.push(root_node)
    while not q.isEmpty() and time.time() - start <= 0.8:
      curr_node = q.pop()
      if curr_node.depth == 900:
          continue

      actions = ['North', 'South', 'East', 'West']
      for action in actions:
        dx, dy = Actions.directionToVector(action)
        child_myPos = (curr_node.myPos[0]+dx, curr_node.myPos[1]+dy)

        if child_myPos in gameState.data.layout.walls.asList():
          continue

        child_enemiesPos = []
        for x,y in curr_node.enemiesPos:
          possibleEnemiesPos = [ (x+dx,y+dy) for (dx, dy) in [(0,1), (0,-1), (1,0), (-1,0), (0,0)] ]
          possibleEnemiesPos = [pos for pos in possibleEnemiesPos if pos not in gameState.data.layout.walls.asList()]
          nearestEnemiesPos = self.myMin([(agent.getMazeDistance(child_myPos, enemiesPos), enemiesPos) for enemiesPos in possibleEnemiesPos])
          for enemy in nearestEnemiesPos:
            if enemy not in child_enemiesPos:
              child_enemiesPos.append(enemy)

        child_node = TreeNode(child_myPos, child_enemiesPos, curr_node)

        if child_myPos in child_node.foodList:
          child_node.foodList.remove(child_myPos)
          child_node.carrying += 1
        elif child_myPos in child_node.capsules:
          child_node.capsules.remove(child_myPos)
          child_node.capsuleCarrying += 1
        child_node.value = evaluate_fun(gameState, child_node)
        curr_node.childs[action] = child_node

        if child_node.myPos not in child_node.enemiesPos or child_node.myPos not in agent.getBorders(gameState):
          q.push(child_node)

    root_node = self.backPropagation(root_node)

    available_actions = []
    for action in root_node.childs:
      child = root_node.childs[action]
      if child!=None and child.value!=-10000:
        available_actions.append(action)
        # print action, child.branchDepth, child.value

    epsilon = 0.9
    if random.randint(0,100)/100 >= 0.9:
      return random.choice(available_actions)
    else:
      return root_node.bestAction

  def myMin(self, listOfTupple):
    key, value = min(listOfTupple)
    return [value for (k,v) in listOfTupple if k==key]

  def backPropagation(self, node):
    if node.isLeafNode():
      return node

    maxValue = -10000
    miniDepth = 10000
    for action in node.childs:
      child = node.childs[action]
      if child!=None:
        child = self.backPropagation(child)
        if child.value > maxValue:
          maxValue = child.value
          miniDepth = child.branchDepth
          bestAction = action
        elif child.value == maxValue and child.branchDepth < miniDepth:
          miniDepth = child.branchDepth
          bestAction = action

    updated_node = node
    updated_node.value = maxValue
    updated_node.branchDepth = miniDepth
    updated_node.bestAction = bestAction
    return updated_node

class TreeNode:

  def __init__(self, myPos, enemiesPos, parent = None):
    if parent == None:
      self.depth = 1
      self.branchDepth = 1
      self.foodList = []
      self.capsules = [] 
      self.carrying = 0
      self.capsuleCarrying = 0
    else:
      self.depth = parent.depth + 1
      self.branchDepth = parent.branchDepth + 1
      self.foodList = parent.foodList
      self.capsules = parent.capsules
      self.carrying = parent.carrying
      self.capsuleCarrying = parent.capsuleCarrying
    self.myPos = myPos
    self.enemiesPos = enemiesPos
    self.parent = parent
    self.childs = {"North":None, "South":None, "East":None, "West":None, "Stop":None}
    self.bestAction = None
    self.value = None

  def isLeafNode(self):
    n_childs = sum([child==None for child in self.childs.values()])
    return n_childs==5


######################################
# Method 2 - Heuristic Search Agents #
######################################

class BasicAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        # Get the x-axis value of the middle wall in the game board
        self.midWidth = gameState.data.layout.width/2
        # Get the legal positions that agents could possibly be.
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        # Use a maze distance calculator
        self.distancer.getMazeDistances()
        # Define which mode the agent is in
        self.offenseTactic = False
        # Get enemies' index
        self.enemyIndices = self.getOpponents(gameState)
        # Initialize position probability distribution of each enemy by assigning every legal position the same probability 
        self.enemyPositionitionDistribution = {}
        for enemy in self.enemyIndices:
            self.enemyPositionitionDistribution[enemy] = util.Counter()
            self.enemyPositionitionDistribution[enemy][gameState.getInitialAgentPosition(enemy)] = 1.


    def initializeEnemyPositionitionDistribution(self, enemy):
        # Initialize position probability distribution of a specified enemy by assigning every legal position the same probability
        self.enemyPositionitionDistribution[enemy] = util.Counter()
        for position in self.legalPositions:
            self.enemyPositionitionDistribution[enemy][position] = 1.0
        self.enemyPositionitionDistribution[enemy].normalize()


    def forwardStep(self, enemy, gameState):
        # Calculate position probability distribution for next step and update
        newDistribution = util.Counter()
        for thisPosition in self.legalPositions:
            nextPositionDistribution = util.Counter()
            possiblePositions = [(thisPosition[0]+1, thisPosition[1]), (thisPosition[0]-1, thisPosition[1]), (thisPosition[0], thisPosition[1]+1), (thisPosition[0], thisPosition[1]-1)]
            for position in possiblePositions:
                if position in self.legalPositions:
                    nextPositionDistribution[position] = 1.0
                else:
                    pass
            nextPositionDistribution.normalize()
            for nextPosition, probability in nextPositionDistribution.items():
                newDistribution[nextPosition] += probability * self.enemyPositionitionDistribution[enemy][thisPosition]
        newDistribution.normalize()
        self.enemyPositionitionDistribution[enemy] = newDistribution


    def observe(self, enemy, observation, gameState):
        # Calculate the position probability distribution for specified enemy
        observedDistance = observation[enemy]
        myPosition = gameState.getAgentPosition(self.index)
        newDistribution = util.Counter()
        for position in self.legalPositions:
            trueDistance = util.manhattanDistance(myPosition, position)
            emissionModel = gameState.getDistanceProb(trueDistance, observedDistance)
            # if self.red:
            #     print self.index, myPosition, position, observedDistance, trueDistance, emissionModel
            if self.red:
                bePacman = position[0] < self.midWidth
            else:
                bePacman = position[0] > self.midWidth
            # If distance is less than 5, we can observe it
            if trueDistance <= 5:
                newDistribution[position] = 0.
            elif bePacman != gameState.getAgentState(enemy).isPacman:
                newDistribution[position] = 0.
            else:
                # Calculate real probability by multiplying observation probability and emission probability
                newDistribution[position] = self.enemyPositionitionDistribution[enemy][position] * emissionModel
        if newDistribution.totalCount() == 0:
            self.initializeEnemyPositionitionDistribution(enemy)
        else:
            newDistribution.normalize()
            self.enemyPositionitionDistribution[enemy] = newDistribution


    def chooseAction(self, gameState):
        # Choose an action based on evaluated values of next possible states 
        myPosition = gameState.getAgentPosition(self.index)
        observedDistances = gameState.getAgentDistances()
        newState = gameState.deepCopy()
        for enemy in self.enemyIndices:
            enemyPosition = gameState.getAgentPosition(enemy)
            if enemyPosition:
                newDistribution = util.Counter()
                newDistribution[enemyPosition] = 1.0
                self.enemyPositionitionDistribution[enemy] = newDistribution
            else:
                self.forwardStep(enemy, gameState)
                self.observe(enemy, observedDistances, gameState)
        for enemy in self.enemyIndices:
            probablePosition = self.enemyPositionitionDistribution[enemy].argMax()
            conf = game.Configuration(probablePosition, Directions.STOP)
            newState.data.agentStates[enemy] = game.AgentState(conf, newState.isRed(probablePosition) != newState.isOnRedTeam(enemy))
        action = self.getMaxEvaluatedAction(newState, depth=2)[1]
        return action


    def getMaxEvaluatedAction(self, gameState, depth):
        # Evaluate state by averaging all possible next states' value
        if depth == 0 or gameState.isOver():
            return self.evluateState(gameState), Directions.STOP
        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        nextGameStates = [gameState.generateSuccessor(self.index, action)
                                 for action in actions] 
        scores = []
        for state in nextGameStates:
            newStates = [state]
            for enemy in self.enemyIndices:
                tmp = [state.generateSuccessor(enemy, action) for action in state.getLegalActions(enemy) for state in newStates]
                newStates = tmp
            nextScores = [self.getMaxEvaluatedAction(state, depth - 1)[0] for state in newStates]
            scores.append(sum(nextScores)/len(nextScores))
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if
                         scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return bestScore, actions[chosenIndex]


    def getEnemyDistances(self, gameState):
        # Return all enemies' positions
        distances = []
        for enemy in self.enemyIndices:
            enemyPosition = gameState.getAgentPosition(enemy)
            myPosition = gameState.getAgentPosition(self.index)
            if enemyPosition:
                pass
            else:
                enemyPosition = self.enemyPositionitionDistribution[enemy].argMax()
            distances.append((enemy, self.distancer.getDistance(myPosition, enemyPosition)))
        return distances


    def evluateState(self, gameState):
        """
        Evaluate the utility of a game state.
        """
        util.raiseNotDefined()


class OffensiveAgent(BasicAgent):
    # An offensive agent have more probability to eat enemies' food

    def registerInitialState(self, gameState):
        BasicAgent.registerInitialState(self, gameState)
        # Define weights for features
        self.returning = False
        self.weights = util.Counter()
        self.weights['score'] = 2
        self.weights['numFood'] = -100
        self.weights['numCapsule'] = -10000
        self.weights['closetDistanceToFood'] = -3
        self.weights['closetDistanceToMiddle'] = -2
        # self.weights['closetDistanceToGhosts'] = 500
        self.weights['closetDistanceToGhosts'] = 100
        self.weights['closetDistanceToCapsule'] = -5

    def chooseAction(self, gameState):
        enemyScareTime = [gameState.getAgentState(enemy).scaredTimer for enemy in self.enemyIndices]
        score = self.getScore(gameState)
        if score < 7:
            carryLimit = 6
        else:
            carryLimit = 4
        if gameState.getAgentState(self.index).numCarrying < carryLimit and len(self.getFood(gameState).asList()) > 2:
            self.returning = False
        else:
            if min(enemyScareTime) > 5:
                self.returning = False
            else:
                self.returning = True
        return BasicAgent.chooseAction(self, gameState)


    def evluateState(self, gameState):
        features = self.getFeatures(gameState)
        return features * self.weights
        # if self.returning:
        #     return - 2 * features['closetDistanceToMiddle'] + 500 * features['closetDistanceToGhosts']
        # else:
        #     if features['minEnemyScareTime'] <= 6 and features['closetDistanceToGhosts'] < 4:
        #         features['closetDistanceToGhosts'] *= -1
        #     return 2 * features['score'] - 100 * features['numFood'] - \
        #            3 * features['closetDistanceToFood'] - 10000 * features['numCapsule']- \
        #            5 * features['closetDistanceToCapsule'] + 100 * features['closetDistanceToGhosts']

    def getFeatures(self, gameState):
        if self.returning:
            myPosition = gameState.getAgentPosition(self.index)
            closetDistanceToMiddle = min([self.distancer.getDistance(myPosition, (self.midWidth, i))
                                 for i in range(gameState.data.layout.height)
                                 if (self.midWidth, i) in self.legalPositions])
            distanceToEnemies = self.getEnemyDistances(gameState)
            distanceToGhosts = [distance for index, distance in distanceToEnemies if
                                 not gameState.getAgentState(index).isPacman]
            closetDistanceToGhosts = min(distanceToGhosts) if len(distanceToGhosts) else 0
            features = util.Counter()
            features['closetDistanceToGhosts'] = closetDistanceToGhosts
            features['closetDistanceToMiddle'] = closetDistanceToMiddle
            return features
        else:
            myPosition = gameState.getAgentPosition(self.index)
            closetDistanceToMiddle = min([self.distancer.getDistance(myPosition, (self.midWidth, i))
                                     for i in range(gameState.data.layout.height)
                                     if (self.midWidth, i) in self.legalPositions])
            foodToEat = self.getFood(gameState).asList()
            distanceToFood = [self.distancer.getDistance(myPosition, food) for
                                 food in foodToEat]
            closetDistanceToFood = min(distanceToFood) if len(distanceToFood) else 0            
            distanceToEnemies = self.getEnemyDistances(gameState)
            distanceToGhosts = [distance for index, distance in distanceToEnemies if
                                 not gameState.getAgentState(index).isPacman]
            closetDistanceToGhosts = min(distanceToGhosts) if len(distanceToGhosts) else 0
            if closetDistanceToGhosts >= 4:
                closetDistanceToGhosts = 0      
            if self.red:
                capsuleToEat = gameState.getBlueCapsules()
            else:
                capsuleToEat = gameState.getRedCapsules()
            for enemy in self.enemyIndices:
                if not gameState.getAgentState(enemy).isPacman:
                    enemyPosition = gameState.getAgentPosition(enemy)
                    if enemyPosition != None:
                        distanceToGhosts.append(self.distancer.getDistance(myPosition, enemyPosition))
            distanceToCapsule = [self.distancer.getDistance(myPosition, capsule) for capsule in
                                        capsuleToEat]
            closetDistanceToCapsule = min(distanceToCapsule) if len(distanceToCapsule) else 0
            enemyScareTime = [gameState.getAgentState(enemy).scaredTimer for enemy
                                 in self.enemyIndices]
            minEnemyScareTime = min(enemyScareTime)
            if minEnemyScareTime <= 6 and closetDistanceToGhosts < 4:
                closetDistanceToGhosts *= -1
            features = util.Counter()
            features['numFood'] = len(foodToEat)
            features['score'] = self.getScore(gameState)
            features['numCapsule'] = len(distanceToCapsule)
            features['closetDistanceToFood'] = closetDistanceToFood
            features['closetDistanceToGhosts'] = closetDistanceToGhosts
            features['closetDistanceToCapsule'] = closetDistanceToCapsule
            return features


class DefensiveAgent(BasicAgent):
    # A defensive agent has more probability to defend its food
    def registerInitialState(self, gameState):
        BasicAgent.registerInitialState(self, gameState)
        self.offenseTactic = False
        self.weights = util.Counter()
        self.weights['score'] = 2
        self.weights['numFood'] = 100
        self.weights['numPacman'] = -99999
        self.weights['numCapsule'] = 0
        self.weights['closetDistanceToFood'] = -3
        self.weights['closetDistanceToGhosts'] = 1
        self.weights['closetDistanceToPacmans'] = -10
        self.weights['closetDistanceToCapsule'] = -1


    def chooseAction(self, gameState):
        invaders = [enemy for enemy in self.enemyIndices if
                    gameState.getAgentState(enemy).isPacman]
        enemyScareTime = [gameState.getAgentState(enemy).scaredTimer for enemy in
                         self.enemyIndices]
        if len(invaders) == 0 or min(enemyScareTime) > 8:
            self.offenseTactic = True
        else:
            self.offenseTactic = False
        return BasicAgent.chooseAction(self, gameState)


    def evluateState(self, gameState):
        features = self.getFeatures(gameState)
        return features * self.weights
        # if self.offenseTactic == False:
        #     return -999999 * features['numPacman'] - 10 * features['closetDistanceToPacmans'] - features['closetDistanceToCapsule']
        # else:
        #     return 2 * features['score'] - 100 * features['numFood'] - \
        #            3 * features['closetDistanceToFood'] + features['closetDistanceToGhosts']

    def getFeatures(self, gameState):
        if self.offenseTactic == False:
            myPosition = gameState.getAgentPosition(self.index)
            distanceToEnemies = self.getEnemyDistances(gameState)
            distanceToPacmans = [distance for index, distance in distanceToEnemies if
                        gameState.getAgentState(index).isPacman]
            closetDistanceToPacmans = min(distanceToPacmans) if len(distanceToPacmans) else 0
            capsuleToDefend = self.getCapsulesYouAreDefending(gameState)
            distanceToCapsule = [self.getMazeDistance(myPosition, capsule) for capsule in
                        capsuleToDefend]
            closetDistanceToCapsule = min(distanceToCapsule) if len(distanceToCapsule) else 0
            features = util.Counter()
            features['numPacman'] = len(distanceToPacmans)
            features['closetDistanceToPacmans'] = closetDistanceToPacmans
            features['closetDistanceToCapsule'] = closetDistanceToCapsule
            return features
        else:
            myPosition = gameState.getAgentPosition(self.index)
            
            distanceToEnemies = self.getEnemyDistances(gameState)
            distanceToGhosts = [distance for index, distance in distanceToEnemies if
                                 not gameState.getAgentState(index).isPacman]
            closetDistanceToGhosts = min(distanceToGhosts) if len(distanceToGhosts) else 0
            
            foodToEat = self.getFood(gameState).asList()
            distanceToFood = [self.distancer.getDistance(myPosition, food) for food in
                             foodToEat]
            closetDistanceToFood = min(distanceToFood) if len(distanceToFood) else 0            
            features = util.Counter()
            features['numFood'] = len(foodToEat)
            features['score'] = self.getScore(gameState)
            features['numGhost'] = len(distanceToGhosts)
            features['closetDistanceToFood'] = closetDistanceToFood
            features['closetDistanceToGhosts'] = closetDistanceToGhosts
            return features



######################################
# Method 3 - MCTS Search Agents #
######################################

class Actions():

    def getSuccessor(self, gameState, action):
        suc = gameState.generateSuccessor(self.index, action)
        pos = suc.getAgentState(self.index).getPosition()
        return suc.generateSuccessor(self.index, action) if pos != nearestPoint(pos) else suc

    def evaluate(self, gameState, action):
        return self.getFeatures(gameState, action) * self.getWeights(gameState, action)

    def getFeatures(self, gameState, action):
        fea = util.Counter()
        suc = self.getSuccessor(gameState, action)
        fea['sucSorce'] = self.agent.getScore(suc)
        return fea

    def getWeights(self, gameState, action):
        return {'sucSorce': 1}


class AttackActions(Actions):
    def __init__(self, agent, index, gameState):
        self.agent = agent
        self.index = index
        self.agent.distancer.getMazeDistances()
        self.retreat = False
        self.numEnemyFood = float("+inf")
        self.counter = 0;
        self.areaWidth = gameState.data.layout.width - 2
        self.boundary = []
        self.patrolSpot = []

        if self.agent.red:
            middle = self.areaWidth // 2
        else:
            middle = self.areaWidth // 2 + 1

        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(middle, i):
                self.boundary.append((middle, i))
        
        while len(self.patrolSpot) > (gameState.data.layout.height - 2) // 2:
            self.patrolSpot.pop(0)
            self.patrolSpot.pop()

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        curPos = successor.getAgentState(self.index).getPosition()
        foodList = self.agent.getFood(successor).asList()
        capsuleList = self.agent.getCapsules(successor)
        minBoundary = 100000

        features['sucSorce'] = self.agent.getScore(successor)
        features['carryingScore'] = successor.getAgentState(self.index).numCarrying
                
        for bd in self.boundary:
            bdDis = self.agent.getMazeDistance(curPos, bd)
            if bdDis < minBoundary:
                minBoundary = bdDis
        features['backhome'] = minBoundary        
        
        if len(foodList) > 0 :
            minFoodDistance = 100000
            for food in foodList:
                distance = self.agent.getMazeDistance(curPos, food)
                if (distance < minFoodDistance):
                    minFoodDistance = distance
            features['foodDis'] = minFoodDistance

        if len(capsuleList) > 0:
            minCapsuleDistance = 100000
            for c in capsuleList:
                distance = self.agent.getMazeDistance(curPos, c)
                if distance < minCapsuleDistance:
                    minCapsuleDistance = distance
            features['capDis'] = minCapsuleDistance
        else:
            features['capDis'] = 0

        enemyState = [successor.getAgentState(i) for i in self.agent.getOpponents(successor)]
        enemyGhost = filter(lambda x: not x.isPacman and x.getPosition() != None, enemyState)
        enemyGhost = list(enemyGhost)
        if len(enemyGhost) > 0:
            positions = [agent.getPosition() for agent in enemyGhost]
            closest = min(positions, key=lambda x: self.agent.getMazeDistance(curPos, x))
            closestDist = self.agent.getMazeDistance(curPos, closest)
            if closestDist <= 5:
                features['GhoDis'] = closestDist

        else:
            probDist = []
            for i in self.agent.getOpponents(successor):
                probDist.append(successor.getAgentDistances()[i])
            features['GhoDis'] = min(probDist)

        
        enemies = [successor.getAgentState(i) for i in self.agent.getOpponents(successor)]
        enemiesPacMan = filter(lambda x: x.isPacman and x.getPosition() != None, enemies)
        enemiesPacMan = list(enemiesPacMan)
        if len(enemiesPacMan) > 0:
            positions = [agent.getPosition() for agent in enemiesPacMan]
            closest = min(positions, key=lambda x: self.agent.getMazeDistance(curPos, x))
            closestDist = self.agent.getMazeDistance(curPos, closest)
            if closestDist < 4:
                features['enemyDis'] = closestDist
        else:
            features['enemyDis'] = 0

        return features

    def getWeights(self, gameState, action):

        successor = self.getSuccessor(gameState, action)
        pointsCarry = successor.getAgentState(self.index).numCarrying
        enemyState = [successor.getAgentState(i) for i in self.agent.getOpponents(successor)]
        enemyGhost = filter(lambda x: not x.isPacman and x.getPosition() != None, enemyState)
        enemyGhost = list(enemyGhost)
        if len(enemyGhost) > 0:
            for agent in enemyGhost:
                if agent.scaredTimer > 0:
                    if agent.scaredTimer > 2*10+2:
                        return {'backhome': 10-3*pointsCarry, 'sucSorce': 2*50+10, 'foodDis': -2*5, 'enemyDis': 0, 'GhoDis': -1, 'capDis': 0, 'carryingScore': 3*100+2*25}
                    elif 2*3 < agent.scaredTimer < 20//10+10 :
                        return {'backhome': -5-4*pointsCarry, 'sucSorce': 2*50+10+5*pointsCarry, 'foodDis': -5, 'enemyDis': 0, 'GhoDis': -1, 'capDis': -10, 'carryingScore': 4*50}
                else:
                    return {'backhome': -1-4*pointsCarry, 'sucSorce': 2*50+10, 'foodDis': -5, 'enemyDis': 0, 'GhoDis': 3*10+5, 'capDis': -2*5, 'carryingScore': 0}
        self.counter += 1
        return {'backhome': 5-3*pointsCarry, 'sucSorce': 999+pointsCarry*3.5, 'foodDis': -7, 'enemyDis': 0, 'GhoDis': 0, 'capDis': -5, 'carryingScore': 3*100+2*25}

    def Simulation(self, gameState, depth, decay):
        new_state = gameState.deepCopy()

        if depth == 0:
            res = []
            actions = new_state.getLegalActions(self.index)
            actions.remove(Directions.STOP)

            reversed_direction = Directions.REVERSE[new_state.getAgentState(self.index).configuration.direction]
            if reversed_direction in actions and len(actions) > 1:
                actions.remove(reversed_direction)
            possible_act = random.choice(actions)
            next_state = new_state.generateSuccessor(self.index, possible_act)
            res.append(self.evaluate(next_state, Directions.STOP))
            return max(res)

        res = []
        actions = new_state.getLegalActions(self.index)
        current_direction = new_state.getAgentState(self.index).configuration.direction

        reversed_direction = Directions.REVERSE[current_direction]
        if reversed_direction in actions and len(actions) > 1:
            actions.remove(reversed_direction)

        for a in actions:
            next_state = new_state.generateSuccessor(self.index, a)
            val = self.evaluate(next_state, Directions.STOP) + decay * self.Simulation(next_state, depth-1, decay)
            res.append(val)
        return max(res)


    def MCTS(self, gameState, depth, decay):

        new_state = gameState.deepCopy()
        value = self.evaluate(new_state, Directions.STOP)
        decay_index = 1
        while depth > 0:

            # Get valid actions
            actions = new_state.getLegalActions(self.index)
            # The agent should not stay put in the simulation
            # actions.remove(Directions.STOP)
            current_direction = new_state.getAgentState(self.agent.index).configuration.direction
            # The agent should not use the reverse direction during simulation
            reversed_direction = Directions.REVERSE[new_state.getAgentState(self.agent.index).configuration.direction]
            if reversed_direction in actions and len(actions) > 1:
                actions.remove(reversed_direction)
            # Randomly chooses a valid action
            a = random.choice(actions)
            # Compute new state and update depth
            new_state = new_state.generateSuccessor(self.agent.index, a)
            value = value + decay ** decay_index * self.evaluate(new_state, Directions.STOP)
            depth -= 1
            decay_index += 1

        # Evaluate the final simulation state
        return value

    def chooseAction(self, gameState):

        actions = gameState.getLegalActions(self.agent.index)
        actions.remove(Directions.STOP)
        simulatedVal = []

        for a in actions:
            NextState = gameState.generateSuccessor(self.agent.index, a)
            # Choose Simulation or MCTS
            # value = self.MCTS(NextState, 2, 0.8)
            value = self.Simulation(NextState, 2, 0.8)
            simulatedVal.append(value)

        return random.choice(list(filter(lambda x: x[0] == max(simulatedVal), zip(simulatedVal, actions))))[1]


class DefendActions(Actions):
    # Load the denfensive information
    def __init__(self, agent, index, gameState):
        self.index = index
        self.agent = agent
        self.defenderList = {}
        self.areaWidth = gameState.data.layout.width - 2
        self.patrolGrid = []
        self.target = None
        self.lastObservedFood = None

        if self.agent.red:
            middle = self.areaWidth // 2
        else:
            middle = self.areaWidth // 2 + 1
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(middle, i):
                self.patrolGrid.append((middle, i))

        while len(self.patrolGrid) > (gameState.data.layout.height - 2) // 2:
            self.patrolGrid.pop(0)
            self.patrolGrid.pop()

        self.updateDefendingGrid(gameState)


    def randomlyPatrol(self):

        rd, total = random.random(), 0
        for key,value in self.defenderList.items():
            total += value
            if rd < total: return key


    def updateDefendingGrid(self, gameState):
        sumProb = 0
        for pos in self.patrolGrid:
            minDis = 100000
            foodList = self.agent.getFoodYouAreDefending(gameState).asList()
            for food in foodList:
                dist = self.agent.getMazeDistance(pos, food)
                if dist < minDis:
                    minDis = dist

            minDis = 1 if not minDis else minDis
            self.defenderList[pos] = 1.0 / float(minDis)
            sumProb += self.defenderList[pos]

        # Normalization
        sumProb = 1 if not sumProb else sumProb
        for x in self.defenderList.keys():
            self.defenderList[x] = float(self.defenderList[x]) / float(sumProb)


    def chooseAction(self, gameState):

        DefendingList = self.agent.getFoodYouAreDefending(gameState).asList()
        if self.lastObservedFood and len(self.lastObservedFood) != len(DefendingList):
            self.updateDefendingGrid(gameState)

        curPos = gameState.getAgentPosition(self.index)
        if curPos == self.target:
            self.target = None

        enemyState = []
        for i in self.agent.getOpponents(gameState):
            enemyState.append(gameState.getAgentState(i))
        seenEnemy = list(filter(lambda x:x.isPacman and x.getPosition() != None, enemyState))
        if len(seenEnemy) > 0:
            enPos = []
            for en in seenEnemy:
                enPos.append(en.getPosition())

            self.target = min(enPos, key=lambda x: self.agent.getMazeDistance(curPos, x))

        elif self.lastObservedFood != None:
            eaten = set(self.lastObservedFood) - set(self.agent.getFoodYouAreDefending(gameState).asList())
            if len(eaten) > 0:
                self.target = eaten.pop()

        self.lastObservedFood = self.agent.getFoodYouAreDefending(gameState).asList()

        if self.target == None and len(self.agent.getFoodYouAreDefending(gameState).asList()) <= 6:
            pacdots = self.agent.getFoodYouAreDefending(gameState).asList() + self.agent.getCapsulesYouAreDefending(gameState)
            self.target = random.choice(pacdots)

        elif self.target == None:
            self.target = self.randomlyPatrol()

        actions = gameState.getLegalActions(self.index)
        act, dis = [], []
        for a in actions:
            new_state = gameState.generateSuccessor(self.index, a)
            if a != Directions.STOP and not new_state.getAgentState(self.index).isPacman:
                newPosition = new_state.getAgentPosition(self.index)
                act.append(a)
                dis.append(self.agent.getMazeDistance(newPosition, self.target))

        return random.choice(list(filter(lambda x: x[0] == min(dis), zip(dis, act))))[1]


class Attacker(CaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.DefenceStatus = DefendActions(self, self.index, gameState)
        self.OffenceStatus = AttackActions(self, self.index, gameState)

    def chooseAction(self, gameState):

        self.enemies = self.getOpponents(gameState)

        return self.DefenceStatus.chooseAction(gameState) if self.getScore(gameState) > 2*10 else self.OffenceStatus.chooseAction(gameState)


class Defender(CaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.DefenceStatus = DefendActions(self, self.index, gameState)
        self.OffenceStatus = AttackActions(self, self.index, gameState)

    def chooseAction(self, gameState):

        self.enemies = self.getOpponents(gameState)

        return self.DefenceStatus.chooseAction(gameState)