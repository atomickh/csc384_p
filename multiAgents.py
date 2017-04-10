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
import datetime
import math

from game import Agent

# add the function scoreEvaluationFunction to multiAgents.py
def scoreEvaluationFunction(currentGameState):
   """
     This default evaluation function just returns the score of the state.
     The score is the same one displayed in the Pacman GUI.

     This evaluation function is meant for use with adversarial search agents
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

     Note: this is an abstract class: one that should not be instantiated.  It's only partially specified, and designed to be extended.  Agent (game.py)
     is another abstract class.
   """

   def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
      self.index = 0 # Pacman is always agent index 0
      self.evaluationFunction = util.lookup(evalFn, globals())
      self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
   """
     Your minimax agent (question 2)
   """
   
   def getAction(self, gameState):
      """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
          Returns a list of legal actions for an agent
          agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
          Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
          Returns the total number of agents in the game
      """
      "*** YOUR CODE HERE ***"
      return self.MinimaxSearch(gameState, 0, 0 )
   
   def MinimaxSearch(self, gameState, curr_depth, agent_index):
      #Check if depth is reached
      if curr_depth >= self.depth or gameState.isLose() or gameState.isWin():
         return scoreEvaluationFunction(gameState)
      
      next_index = agent_index + 1
      next_depth = curr_depth
      if next_index >= gameState.getNumAgents():
         next_index = 0
         next_depth += 1
      
      action_outcomes = []
      action_list = gameState.getLegalActions(agent_index)
      if 'Stop' in action_list:
         action_list.remove('Stop')
      for action in action_list:
            
         successor = gameState.generateSuccessor(agent_index, action)
         action_outcomes.append(self.MinimaxSearch(successor, next_depth, next_index))
         
      
      if agent_index == 0:
         if curr_depth == 0:
            #first level
            choice_val = max(action_outcomes)
            choice = action_outcomes.index(choice_val)
            return action_list[choice]
         else:
            return max(action_outcomes)
      else:
         return min(action_outcomes)
      
         
      
         

   
class AlphaBetaAgent(MultiAgentSearchAgent):
   """
     Your minimax agent with alpha-beta pruning (question 3)
   """

   def getAction(self, gameState):
      """
        Returns the minimax action using self.depth and self.evaluationFunction
      """
      "*** YOUR CODE HERE ***"
      #util.raiseNotDefined()
      return self.AlphaBetaSearch(gameState, 0, 0, -1e8, 1e8 )
   
   def AlphaBetaSearch(self, gameState, curr_depth, agent_index, alpha, beta):
      #Check if depth is reached
      if curr_depth >= self.depth or gameState.isLose() or gameState.isWin():
         return scoreEvaluationFunction(gameState)
      
      next_index = agent_index + 1
      next_depth = curr_depth
      if next_index >= gameState.getNumAgents():
         next_index = 0
         next_depth += 1
      
      action_outcomes = []
      action_list = gameState.getLegalActions(agent_index)
      if 'Stop' in action_list:
         action_list.remove('Stop')
      
      best_choice = '' 
      if agent_index == 0:
         for action in action_list:
            successor = gameState.generateSuccessor(agent_index, action)
            
            successor_res = self.AlphaBetaSearch(successor, next_depth, next_index, alpha, beta)
            
            if alpha < successor_res:
               alpha = successor_res 
               best_choice = action
               if beta <= alpha:
                  break
         if curr_depth == 0:
            return best_choice
         return alpha
      else:
         for action in action_list:
            successor = gameState.generateSuccessor(agent_index, action)
            
            successor_res = self.AlphaBetaSearch(successor, next_depth, next_index, alpha, beta)
            
            if beta > successor_res:
               beta = successor_res 
               best_choice = action
               if beta <= alpha:
                  break
         return beta
      
      
      
class ExpectimaxAgent(MultiAgentSearchAgent):
   """
     Your expectimax agent (question 4)
   """

   def getAction(self, gameState):
      """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
      """
      "*** YOUR CODE HERE ***"
      #util.raiseNotDefined()
      return self.ExpectimaxSearch(gameState, 0, 0)
      
      
   def ExpectimaxSearch(self, gameState, curr_depth, agent_index):
      #Check if depth is reached
      if curr_depth >= self.depth or gameState.isLose() or gameState.isWin():
         return scoreEvaluationFunction(gameState)
      
      next_index = agent_index + 1
      next_depth = curr_depth
      if next_index >= gameState.getNumAgents():
         next_index = 0
         next_depth += 1
      
      action_outcomes = []
      action_list = gameState.getLegalActions(agent_index)
      if 'Stop' in action_list:
         action_list.remove('Stop')
      for action in action_list:
            
         successor = gameState.generateSuccessor(agent_index, action)
         action_outcomes.append(self.ExpectimaxSearch(successor, next_depth, next_index))
         
      
      if agent_index == 0:
         if curr_depth == 0:
            #first level
            choice_val = max(action_outcomes)
            choice = action_outcomes.index(choice_val)
            return action_list[choice]
         else:
            return max(action_outcomes)
      else:
         total_expect = sum(action_outcomes)
         return total_expect/len(action_outcomes)
        
        
class MonteCarloAgent(MultiAgentSearchAgent):
   """
       Your monte-carlo agent (question 5)
       ***UCT = MCTS + UBC1***
       TODO:
       1) Complete getAction to return the best action based on UCT.
       2) Complete runSimulation to simulate moves using UCT.
       3) Complete final, which updates the value of each of the states visited during a play of the game.

       * If you want to add more functions to further modularize your implementation, feel free to.
       * Make sure that your dictionaries are implemented in the following way:
           -> Keys are game states.
           -> Value are integers. When performing division (i.e. wins/plays) don't forget to convert to float.
     """

   def __init__(self, evalFn='mctsEvalFunction', depth='-1', timeout='50', numTraining=100, C='2', Q=None):
      # This is where you set C, the depth, and the evaluation function for the section "Enhancements for MCTS agent".
      if Q:

         
         if Q == 'minimaxClassic':
            evalFn = 'scoreEvaluationFunction'
            depth = -1
            C='6'
            pass
         elif Q == 'testClassic':
            evalFn = 'scoreEvaluationFunction'
            depth = 4
            C = '2'
            pass
         elif Q == 'smallClassic':
            evalFn = 'scoreEvaluationFunction'
            depth = 1
            C = '5'
            pass
         else: # Q == 'contestClassic'
            evalFn = 'betterEvaluationFunction'
            C = '8'
            depth = 4
            assert( Q == 'contestClassic' )
            pass
      # Otherwise, your agent will default to these values.
      else:
         self.C = int(C)
         # If using depth-limited UCT, need to set a heuristic evaluation function.
         if int(depth) > 0:
            evalFn = 'scoreEvaluationFunction'
      
      self.C = int(C)
      self.states = []
      self.plays = dict()
      self.wins = dict()
      self.calculation_time = datetime.timedelta(milliseconds=int(timeout))

      self.numTraining = numTraining

      "*** YOUR CODE HERE ***"

      MultiAgentSearchAgent.__init__(self, evalFn, depth)

   def update(self, state):
      """
      You do not need to modify this function. This function is called every time an agent makes a move.
      """
      self.states.append(state)
      
   def best_choice(self, state):
      
      action_list  = state.getLegalActions(0)
      if 'Stop' in action_list and len(action_list)>1:
         action_list.remove('Stop')

      child_list = [(act, state.generateSuccessor(0, act)) for act in action_list]
      
      child_score = [(float(self.wins.get(child, -1))/float(self.plays.get(child, 1)), act) for (act, child) in child_list]
       
      return max(child_score)[1]

      
   def getAction(self, gameState):
      """
      Returns the best action using UCT. Calls runSimulation to update nodes
      in its wins and plays dictionary, and returns best successor of gameState.
      """
      "*** YOUR CODE HERE ***"
      games = 0
      
      acts = gameState.getLegalActions(0)
      if len(acts) == 1:
         return acts[0]
      elif len(acts) == 0:
         return
      
      begin = datetime.datetime.utcnow()
      
      while datetime.datetime.utcnow() - begin < self.calculation_time:
         self.run_simulation(gameState)
         games += 1
     
      return self.best_choice(gameState)
      
   
      
     
   def run_simulation(self, state):
      """
      Simulates moves based on MCTS.
      1) (Selection) While not at a leaf node, traverse tree using UCB1.
      2) (Expansion) When reach a leaf node, expand.
      4) (Simulation) Select random moves until terminal state is reached.
      3) (Backpropapgation) Update all nodes visited in search tree with appropriate values.
      * Remember to limit the depth of the search only in the expansion phase!
      Updates values of appropriate states in search with with evaluation function.
      """
      "*** YOUR CODE HERE ***"
      
      path = []
      index = 0
      curr_state = state
      ret = False
      na = state.getNumAgents()
      expand= True
      depth = 0
      sim = False
      
      if state.isWin() or state.isLose():
         return
      
      while True:
         if index == na and expand == False:
            depth +=1 
         
         if curr_state.isWin() or curr_state.isLose() or depth == self.depth:
            break
            
         index = index % na
         
         sum_plays = 0
         children = []
         
         
         actions = curr_state.getLegalActions(index)
         for next in actions:
            if next == 'Stop':
               continue
            child = curr_state.generateSuccessor(index, next)
            if child not in (self.plays) and expand == True:
               path.append(child)
               self.plays[child] = 0
               self.wins[child] = 0
               expand = False
               curr_state = child
               index += 1
               break
            children.append([next, curr_state, child])
            if expand:
               sum_plays += self.plays[child]
         
         if expand == False and sim == False:
            sim = True
            continue
         
         if expand == True:
            UCB = []
            for (act, parent, child) in children:
               if index == 0:
                  calc = float(self.wins[child])/float(self.plays[child])
                  
               else:
                  calc = 1 - float(self.wins[child])/float(self.plays[child])
               calc += self.C*math.sqrt(2*math.log(sum_plays)/float(self.plays[child]))
               UCB.append((calc, child))
            
            while len(UCB) > 0:
               i = UCB.index(max(UCB))
               c = children[i][2]
               if c.isWin() or c.isLose():
                  path.append(c)
               path.append(c)
               curr_state = c
               index += 1
               break
         
         
            # if len(UCB) == 0:
            #    
            #    if len(path) == 1:
            #       curr_state = state
            #       foo = path.pop[0]
            #    elif len(path) == 0:
            #       return
            #    else:
            #       curr_state = path[-2]
            #       index = index - 2
            #       foo = path.pop[-1]
         else:
            chosen = random.choice(children)
            child = chosen[2]
            curr_state = child
            index += 1
               
      prop_score = self.evaluationFunction(curr_state)
      for s in path:
         self.plays[s] += 1
         self.wins[s] += prop_score
      # End loop
         
      
            
      
         
            


      
   def final(self, state):
      """
      Called by Pacman game at the terminal state.
      Updates search tree values of states that were visited during an actual game of pacman.
      """
      "*** YOUR CODE HERE ***"
      for s in set(self.states):
         if s in self.plays:
            self.plays[s] += 1
            self.wins[s] += self.evaluationFunction(state)
      #pass
     # util.raiseNotDefined()

def mctsEvalFunction(state):
   """
   Evaluates state reached at the end of the expansion phase.
   """
   return 1 if state.isWin() else 0

def betterEvaluationFunction(currentGameState):
   """
     Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
     evaluation function (to help improve your UCT MCTS).
   """
   "*** YOUR CODE HERE ***"
   state = currentGameState
   score = state.getScore()

   pac = state.getPacmanPosition()
   food = state.getFood()
   dist_list = []
   y = 0
   for row in food:
      x = 0
      for node in row:
         if node == True:
            dist = abs(pac[0]-x)+abs(pac[1]-y)
            dist_list.append(dist)
         x+= 1
      y += 1
   if len(dist_list) == 0:
      dist_list = [0]
   #terrain = state.getFood()
   #terrain = str(terrain)
   #food = terrain.count('F')
   #score = 10*(1 - food/len(terrain))
   return score - min(dist_list)
   #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
