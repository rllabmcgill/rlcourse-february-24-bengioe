import networkx as nx
import sys
import numpy
import numpy as np
import matplotlib.pyplot as pp

class MDP:
    @property
    def nstates(self):
        raise NotImplemented()

    @property
    def nactions(self):
        raise NotImplemented()

    def reset(self): pass
    def getState(self): pass # -> int
    def act(self, action): pass # -> reward
    def isGameOver(self): return True




class FourRooms(MDP):
    def __init__(self, size=10):
        self.size = size
        self.allowed_states = []
        self.i2s = {}
        self.s2i = {}
        self.deltas = [[1,0],[0,1],[-1,0],[0,-1]]
        self.s = 0
        self.grid = np.zeros((self.size,self.size))
        for i in range(self.size):
            self.grid[i,0] = 1
            self.grid[0,i] = 1
            self.grid[self.size-1,i] = 1 
            self.grid[i,self.size-1] = 1
            self.grid[self.size/2,i] = 1
            self.grid[i,self.size/2] = 1
        self.grid[self.size/4,self.size/2] = 0
        self.grid[3*self.size/4,self.size/2] = 0
        self.grid[self.size/2,self.size/4] = 0
        self.grid[self.size/2,3*self.size/4] = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i,j] == 0:
                    self.allowed_states.append((i,j))
        self.i2s = dict(enumerate(self.allowed_states))
        self.s2i = dict((s,i) for i,s in self.i2s.iteritems())
        self.resetGoal()
        self.reset()
    def resetGoal(self):
        self.goal = np.random.randint(0,self.nstates)
    def reset(self):
        self.s = np.random.randint(0,self.nstates)
        while self.goal == self.s:
            self.s = np.random.randint(0,self.nstates)
    def viewGridOfS(self, v):
        grid = np.zeros((self.size,self.size))
        for i,(x,y) in enumerate(self.allowed_states):
            grid[x,y] = v[i]
        return grid
    def neighboursOf(self, s):
        ns = []
        pos = self.i2s[s]
        for delta in self.deltas:
            newpos = (pos[0]+delta[0], pos[1]+delta[1])
            if self.grid[newpos[0], newpos[1]] == 0:
                ns.append(self.s2i[newpos])
        return ns
            
            
                
    @property
    def nstates(self):
        return len(self.allowed_states)
    @property
    def nactions(self):
        return 4

    def getState(self):
        return self.s
    def isGameOver(self):
        return self.s == self.goal
    def act(self, action):
        pos = self.i2s[self.s]
        delta = self.deltas[action]
        newpos = (pos[0]+delta[0], pos[1]+delta[1])
        if self.grid[newpos[0], newpos[1]] == 1:
            r = -1 # reward for bumping into a wall
            s = self.s
        else:
            s = self.s2i[newpos]
            r = 1 if s == self.goal else 0
        self.s = s
        return r
    

def q_learning(mdp, gamma=0.9, lr=0.5):
    qsa = np.zeros((mdp.nstates, mdp.nactions)) + .1
    hasConverged = False
    hasReconverged = False
    itr = 0
    nsteps = 0
    TD = 0
    tr = 0
    while not hasConverged or not hasReconverged:
        R = 0
        esteps = 0
        mdp.reset()
        while not mdp.isGameOver():
            s = mdp.getState()
            a = qsa[s].argmax()
            r = mdp.act(a)
            sp = mdp.getState()
            td = r +  gamma * qsa[sp].max() - qsa[s,a]
            qsa[s,a] = qsa[s,a] + lr *(td)
            R += r
            TD += abs(td)
            nsteps += 1
            esteps += 1
        if not itr % 100:
            print itr, nsteps, TD, R, esteps,' '*50,'\b'*100,
            sys.stdout.flush()
            if TD < 0.01:
                if not hasConverged:
                    hasConverged = True
                    mdp.resetGoal()
                    print
                    print 'converged in', itr,'episodes and',nsteps,'steps',tr
                    nconv = nsteps
                else:
                    hasReconverged = True
                    nreconv = nsteps-nconv
            TD = 0
        tr += R
        itr += 1
    print
    print 'reconverged in', itr,'episodes and',nsteps,'steps',tr
    print nconv,nreconv
    return qsa

def subgoals_options(mdp, subgoals='random', nbottlenecks=5, gamma=0.99, lr=0.5, epsilon=0.05):
    """
    define random subgoals, give rewards to options if they reach the subgoal
    exception: option 0 only receives the true reward
    
    """
    if subgoals is 'random':
        topk = np.arange(mdp.nstates)
        np.random.shuffle(topk)
        topk = topk[:nbottlenecks]
    else:
        topk = subgoals
        nbottlenecks = len(subgoals)
    
    qso = np.zeros((mdp.nstates, nbottlenecks)) + 1
    wsa = np.zeros((nbottlenecks, mdp.nstates, mdp.nactions)) + 1
    beta = np.zeros((nbottlenecks, mdp.nstates)) + 0.25
    hasConverged = False
    hasReconverged = False
    itr = 0
    nsteps = 0
    TD = 0
    tr = 0
    while not hasConverged or not hasReconverged:
        R = 0
        esteps = 0
        mdp.reset()
        currentOption = 0
        optionS0 = mdp.getState()
        optionk = 0
        optionReturn = 0
        while not mdp.isGameOver():
            s = mdp.getState()
            a = wsa[currentOption, s].argmax()
            if numpy.random.random() < epsilon:
                a = numpy.random.randint(0, mdp.nactions)
            r = mdp.act(a)
            sp = mdp.getState()
            optionReturn = optionReturn * gamma + r
            for o in range(nbottlenecks):
                # intra option q learning:
                #U = (1-beta[currentOption, sp]) * qso[s,o] + beta[currentOption, sp] * qso[s].max()
                #otd = r + gamma * U - qso[s,o]
                #qso[s,o] = qso[s,o] + lr * otd
                
                #optR = 0
                if o > 0 and s == topk[o]:
                    optR = 1
                else:
                    optR = r
                if wsa[o,s].argmax() == a:
                    td = optR + gamma * wsa[o, sp].max() - wsa[o,s,a]
                    wsa[o,s,a] = wsa[o,s,a] + lr *(td)
                    TD += abs(td)
            R += r
            nsteps += 1
            esteps += 1
            optionk += 1
            if numpy.random.random() < beta[currentOption, s] or esteps > 1000 or mdp.isGameOver():
                # SMDP learning
                qso[optionS0, currentOption] += lr * (optionReturn + gamma**optionk * qso[sp].max()-qso[optionS0, currentOption])
                optionS0 = sp
                currentOption = qso[s].argmax()
                if numpy.random.random() < epsilon:
                    currentOption = numpy.random.randint(0,nbottlenecks)
                #print "new option",currentOption
            if esteps > 1000:
                break
        
        if not itr % 100:
            print itr, nsteps, TD, R, esteps,mdp.goal,topk,' '*10,'\b'*100,
            sys.stdout.flush()
            if TD < 0.01:
                if not hasConverged:
                    hasConverged = True
                    mdp.resetGoal()
                    qso *= 0
                    qso += 1
                    nconv = nsteps
                    print
                    print 'converged in', itr,'episodes and',nsteps,'steps',tr
                else:
                    hasReconverged = True
                    nreconv = nsteps-nconv
            TD = 0
        tr += R
        itr += 1
    print
    print 'reconverged in', itr,'episodes and',nsteps,'steps',tr
    print nconv, nreconv 
    #pp.matshow(mdp.viewGridOfS(visits))
    #pp.show()


def bottleneck_options(mdp, nbottlenecks, gamma=0.99, lr=0.5, epsilon=0.05):
    """
    Define 1 option per bottleneck, give reward to option k if it
    reaches the kth most visited state
    """
    qso = np.zeros((mdp.nstates, nbottlenecks)) + 1
    wsa = np.zeros((nbottlenecks, mdp.nstates, mdp.nactions)) + 1
    visits = np.zeros(mdp.nstates)
    beta = np.zeros((nbottlenecks, mdp.nstates)) + 0.25
    topk = np.arange(nbottlenecks)
    
    hasConverged = False
    hasReconverged = False
    nconv = nreconv = 0
    itr = 0
    nsteps = 0
    TD = 0
    tr = 0
    print "exploring bottlenecks..."
    qsa  = q_learning(mdp)
    for j in range(1):
        for i in range(mdp.nstates):
            r = 0
            mdp.reset()
            mdp.s = i
            esteps = 0
            episode = []
            while not mdp.isGameOver():
                s = mdp.getState()
                a = qsa[s].argmax()
                r = mdp.act(a)
                sp = mdp.getState()
                episode.append(sp)
                if esteps > 1000:
                    break
                esteps += 1
            if r > 0:
                for i in set(episode):
                    visits[i] += 1
    topk = numpy.array(sorted(zip(range(mdp.nstates), visits), key=lambda x:x[1],reverse=True)[:nbottlenecks])[:,0]
    while not hasConverged or not hasReconverged:
        R = 0
        esteps = 0
        mdp.reset()
        episode = []
        currentOption = 0
        optionS0 = mdp.getState()
        optionk = 0
        optionReturn = 0
        while not mdp.isGameOver():
            s = mdp.getState()
            a = wsa[currentOption, s].argmax()
            if numpy.random.random() < epsilon:
                a = numpy.random.randint(0, mdp.nactions)
            r = mdp.act(a)
            sp = mdp.getState()
            episode.append(sp)
            optionReturn = optionReturn * gamma + r

            for o in range(nbottlenecks):
                # intra option q learning
                #U = (1-beta[currentOption, sp]) * qso[s,o] + beta[currentOption, sp] * qso[s].max()
                #otd = r + gamma * U - qso[s,o]
                #qso[s,o] = qso[s,o] + lr * otd
                optR = r
                if s == topk[o]:
                    optR += 1
                if wsa[o,s].argmax() == a:
                    td = optR + gamma * wsa[o, sp].max() - wsa[o,s,a]
                    wsa[o,s,a] = wsa[o,s,a] + lr *(td)
                    TD += abs(td)
            R += r
            nsteps += 1
            esteps += 1
            optionk += 1
            if numpy.random.random() < beta[currentOption, s]:
                # SMDP learning
                qso[optionS0, currentOption] += lr * (optionReturn + gamma**optionk * qso[sp].max()-qso[optionS0, currentOption])
                optionS0 = sp
                currentOption = qso[s].argmax()
            if esteps > 1000:
                break
        if R > 0:
            for i in set(episode):
                visits[i] += 1
        topk = numpy.array(sorted(zip(range(mdp.nstates), visits), key=lambda x:x[1],reverse=True)[:nbottlenecks])[:,0]
        
        if not itr % 100:
            print itr, nsteps, TD, R, esteps,mdp.goal,topk,' '*10,'\b'*200,
            sys.stdout.flush()
            if TD < 0.01:
                if not hasConverged:
                    hasConverged = True
                    mdp.resetGoal()
                    qso *= 0
                    qso += 1
                    visits = np.zeros(mdp.nstates)
                    print
                    print 'converged in', itr,'episodes and',nsteps,'steps',tr
                    nconv = nsteps
                else:
                    hasReconverged = True
                    nreconv = nsteps-nconv
            TD = 0
        tr += R
        itr += 1
    print
    print 'reconverged in', itr,'episodes and',nsteps,'steps',tr
    print nconv, nreconv 




    
def graph_metric_subgoals_options(mdp, metric, gamma=0.99, lr=0.5, epsilon=0.05):
    """
    Define an option for each local minima of the betweenness graph, plus one option
    that does normal q-learning
    """
    G = nx.Graph()
    G.add_nodes_from(range(mdp.nstates))
    for i in range(mdp.nstates):
        for j in mdp.neighboursOf(i):
            G.add_edge(i, j)
    values = metric(G)
    print values
    nx.draw_spring(G, cmap=pp.get_cmap('jet'),
                   node_color=[values[i] for i in G.nodes()],k=1,iterations=5000)
    pp.show()
    pp.matshow(mdp.viewGridOfS(values),cmap='gray')
    pp.show()
    subgoals = [-1]
    for i in range(mdp.nstates):
        if values[i] > max(values[j] for j in mdp.neighboursOf(i)):
            subgoals.append(i)
    print subgoals, len(subgoals)
    subgoals_options(mdp, subgoals=subgoals)
    
print FourRooms().grid
q_learning(FourRooms(10))
subgoals_options(FourRooms(10), 'random', 10)
bottleneck_options(FourRooms(10), 10)
graph_metric_subgoals_options(FourRooms(10), nx.betweenness_centrality)
graph_metric_subgoals_options(FourRooms(10), nx.closeness_centrality)
graph_metric_subgoals_options(FourRooms(10), nx.current_flow_closeness_centrality)
def comm(G):
    cm = nx.communicability(G)
    return dict((i, np.mean(cm[i].values())) for i in cm)
graph_metric_subgoals_options(FourRooms(10), comm)
