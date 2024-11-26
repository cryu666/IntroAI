# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        return childGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (par1-1)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        
        # for pacman agent
        def maxLevel(gameState, depth):
            
            currDepth = depth + 1
            
            # if reach a terminal state, return score and no action
            if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
                return self.evaluationFunction(gameState)
            
            maxvalue = -999999
            actions = gameState.getLegalActions(0)   # agentIndex=0 means pacman, ghosts are >= 1
            for action in actions:
                successor = gameState.getNextState(0, action)
                maxvalue = max(maxvalue, minLevel(successor, currDepth, 1))
                
            return maxvalue
        
        # for all ghosts
        def minLevel(gameState, depth, agentIndex):
            
            # if reach a terminal state, return score and no action
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            minvalue = 999999
            actions = gameState.getLegalActions(agentIndex)  
            
            # calculate score for each possible action bt recursively calling minLevel
            for action in actions:
                successor = gameState.getNextState(agentIndex, action)
                
                # if you are on the last ghost, is it actually pacman, so call maxLevel
                if agentIndex == (gameState.getNumAgents() - 1):
                    minvalue = min(minvalue, maxLevel(successor, depth))
                else:
                    minvalue = min(minvalue, minLevel(successor, depth, agentIndex + 1))
                    
            return minvalue
        
        # root level action
        actions = gameState.getLegalActions(0)
        currentScore = -999999
        returnAction = ''
        
        for action in actions:
            nextState = gameState.getNextState(0, action)
            # next level is a min level. Hence calling minLevel for successors of the root
            score = minLevel(nextState, 0, 1)
            
            # choose the action which is maximum of the successors
            if score > currentScore:
                returnAction = action
                currentScore = score
        return returnAction
        
        
        # util.raiseNotDefined()  
        # End your code

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (part1-2)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        
        # for pacman agent
        def maxLevel(gameState, depth):
            
            currDepth = depth + 1
            
            # if reach a terminal state, return score and no action
            if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
                return self.evaluationFunction(gameState)
            
            maxvalue = -999999
            actions = gameState.getLegalActions(0)   # agentIndex=0 means pacman, ghosts are >= 1
            totalmaxvalue = 0
            numberofactions = len(actions)
            
            for action in actions:
                successor = gameState.getNextState(0, action)
                maxvalue = max(maxvalue, expectLevel(successor, currDepth, 1))
                
            return maxvalue
        
        
        # for all ghosts
        def expectLevel(gameState, depth, agentIndex):
            
            # if reach a terminal state, return score and no action
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            actions = gameState.getLegalActions(agentIndex)  
            totalexpectvalue = 0
            numberofactions = len(actions)
            
            # calculate score for each possible action bt recursively calling minLevel
            for action in actions:
                successor = gameState.getNextState(agentIndex, action)
                
                # if you are on the last ghost, is it actually pacman, so call maxLevel
                if agentIndex == (gameState.getNumAgents() - 1):
                    expectvalue = maxLevel(successor, depth)
                else:
                    expectvalue = expectLevel(successor, depth, agentIndex+1)
                totalexpectvalue = totalexpectvalue + expectvalue
                
            if numberofactions == 0:
                return 0
              
            return float(totalexpectvalue)/float(numberofactions)
        
        # root level action
        actions = gameState.getLegalActions(0)
        currentScore = -999999
        returnAction = ''
        
        for action in actions:
            nextState = gameState.getNextState(0, action)
            # next level is a expect level. Hence calling expectLevel for successors of the root
            score = expectLevel(nextState, 0, 1)
            
            # choose the action which is maximum of the successors
            if score > currentScore:
                returnAction = action
                currentScore = score
        return returnAction
        
        # util.raiseNotDefined()  
        # End your code

better = scoreEvaluationFunction
