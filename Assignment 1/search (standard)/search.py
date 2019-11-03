# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [n, s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE IF YOU WANT TO PRACTICE ***"
    stack = util.Stack()
    visited = []
    stack.push((problem.getStartState(), []))

    while not stack.isEmpty():
        cur_node, actions = stack.pop()
        if problem.isGoalState(cur_node):
            return actions
 
        if cur_node not in visited:
            nextnode = problem.getSuccessors(cur_node)
            visited.append(cur_node)
            for location, direction, cost in nextnode:
                if (location not in visited):
                    stack.push((location, actions + [direction]))

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE IF YOU WANT TO PRACTICE ***"
    queue = util.Queue()
    visited = []
    queue.push((problem.getStartState(), []))

    while not queue.isEmpty():
        cur_node, actions = queue.pop()
        if problem.isGoalState(cur_node):
            return actions
 
        if cur_node not in visited:
            nextnode = problem.getSuccessors(cur_node)
            visited.append(cur_node)
            for location, direction, cost in nextnode:
                if (location not in visited):
                    queue.push((location, actions + [direction]))

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE IF YOU WANT TO PRACTICE ***"
    prior_queue = util.PriorityQueueWithFunction(lambda x: x[2])
    prior_queue.push((problem.getStartState(), None, 0))
    visited = []
    path = {}
    path[(problem.getStartState(), None)] = None
    while not prior_queue.isEmpty():
        cur_fullstate = prior_queue.pop()
        if (problem.isGoalState(cur_fullstate[0])):
            break
        else:
            cur_state = cur_fullstate[0]
            if cur_state not in visited:
                visited.append(cur_state)
            else:
                continue
            successors = problem.getSuccessors(cur_state)
            for state in successors:
                cost = cur_fullstate[2] + state[2];
                if state[0] not in visited:
                    prior_queue.push((state[0], state[1], cost))
                    path[(state[0], state[1])] = cur_fullstate
    
    goal = (cur_fullstate[0], cur_fullstate[1])
    res = []
    while True:
        if path[goal] == None:
            break
        res.append(goal[1])
        goal = (path[goal][0], path[goal][1])
    
    return res[::-1]

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE IF YOU WANT TO PRACTICE ***"
    prior_que = util.PriorityQueue()
    prior_que.push((problem.getStartState(), []), 0)
    visited, actions = [], []

    while prior_que:
        cur, actions = prior_que.pop()
        if problem.isGoalState(cur):
            break
        if cur not in visited:
            visited.append(cur)
            nextnode = problem.getSuccessors(cur)
            for successor, action, cost in nextnode:
                tempActions = actions + [action]
                nextCost = problem.getCostOfActions(tempActions) + heuristic(successor, problem)
                if successor not in visited:
                    prior_que.push((successor, tempActions), nextCost)
    return actions

    util.raiseNotDefined()

def iterativeDeepeningSearch(problem):
    """Search the deepest node in an iterative manner."""
    "*** YOUR CODE HERE FOR TASK 1 ***"
    
    def DLS(node, depth, path):
        if depth == 0:
            if problem.isGoalState(node):
                return path + []
        elif depth > 0:
            for suc in problem.getSuccessors(node):
                if suc[0] not in visited:
                    visited.append(suc[0])
                    found = DLS(suc[0], depth-1, path+[suc[1]])
                    if found != None:
                        return found
        return None

    depth = 0
    visited = []

    # total nodes can be explored in theory
    total_nodes = (problem.walls.width - 2) * (problem.walls.height - 2)

    while True:
        if depth > total_nodes:
            print('CAN NOT FIND GOAL')
            return

        res = DLS(problem.getStartState(), depth, [])
        if res != None:
            return res
        depth += 1
        visited = []

    util.raiseNotDefined()

def waStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has has the weighted (x 2) lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE FOR TASK 2 ***"
    '''
    Weighted A* (To solve Breaking Ties)

    h = W * heuristic
    W = (1 + p)

    若 W -> Inf, wA* 表现趋向为 Greedy bset-first search (expand的结点变少 但不一定是最优路径)
    若 W -> 0, wA* 表现趋向为 Uniform cost search (Dijkstra algorithm) (一定是最优路径 但expand的结点变多)

    选择因子 p 使得 p < 移动一步(step)的最小代价 / 期望的最长路径长度
    假设你不希望你的路径超过1000步(step), 你可以使p = 1 / 1000
    添加这个附加值的结果是, 在保证最优路径的情况下A*比以前搜索的结点更少了
    
    此代码设置 W = 2
    (虽然 W = 1.002 基本能找出最短路径 但会expand更多结点更为耗时 W = 2 不一定找到最短路径 但较省时)
    '''
    prior_que = util.PriorityQueue()
    prior_que.push((problem.getStartState(), []), 0)
    visited, actions = [], []

    while prior_que:
        cur, actions = prior_que.pop()
        if problem.isGoalState(cur):
            break
        if cur not in visited:
            visited.append(cur)
            nextnode = problem.getSuccessors(cur)
            for successor, action, cost in nextnode:
                tempActions = actions + [action]
                nextCost = problem.getCostOfActions(tempActions) + 2*heuristic(successor, problem)
                if successor not in visited:
                    prior_que.push((successor, tempActions), nextCost)
    return actions

    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
wastar = waStarSearch
