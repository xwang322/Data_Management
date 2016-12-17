from __future__ import print_function
from graph import Graph
import numpy as np

""" Graphs for testing sum product implementation
"""

def checkEq(a,b):
    epsilon = 10**-6
    return abs(a-b) < epsilon

def makeToyGraph():
    """ Simple graph encoding, basic testing
        2 vars, 2 facs
        f_a, f_ba - p(a)p(a|b)
        factors functions are a little funny but it works
    """
    G = Graph()

    a = G.addVarNode('a', 2)
    b = G.addVarNode('b', 2)
    c = G.addVarNode('c', 2)

    F1 = np.array([[1, 2], [2, 4]])
    G.addFacNode(F1, a, b)

    F2 = np.array([[2, 2], [1, 4]])
    G.addFacNode(F2, b, c)

    return G

def testToyGraph():
    """ Actual test case
    """

    G = makeToyGraph()
    marg = G.marginals()
    brute = G.bruteForce()

    # check the results
    # want to verify incoming messages
    # if vars are correct then factors must be as well
    a = G.var['a'].incoming
    print('a01', a[0][0])
    print('a02', a[0][1])

    b = G.var['b'].incoming
    print('b01', b[0][0])
    print('b02', b[0][1])
    print('b03', b[1][0])
    print('b04', b[1][1])

    c = G.var['c'].incoming
    print('c01', c[0][0])
    print('c02', c[0][1])
	
    # check the marginals
    am = marg['a']
    print('a1', am[0])
    print('a2', am[1])

    bm = marg['b']
    print('b1', bm[0])
    print('b2', bm[1])

    cm = marg['c']
    print('c1', cm[0])
    print('c2', cm[1])
	
    print("All tests passed!")

def makeTestGraph():

    G = Graph()

    a = G.addVarNode('a', 4)
    b = G.addVarNode('b', 4)

    p1 = np.array([[1,1,1,1],[1,1,1,1],[3,3,3,12],[1,1,1,1]])
    G.addFacNode(p1, a, b)

    p2 = np.array([[1,1,1,1],[2,2,6,2],[1,1,1,1],[1,1,1,1]])
    G.addFacNode(p2, a, b)

    p3 = np.array([[1,2,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    G.addFacNode(p3, a, b)

    return G

def testTestGraph():
    """ Automated test case
    """
    G = makeTestGraph()
    marg = G.marginals()
    brute = G.bruteForce()

    # check the marginals
    am = marg['a']
    print('a1', am[0])
    print('a2', am[1])
    print('a3', am[2])
    print('a4', am[3])

    bm = marg['b']
    print('b1', bm[0])
    print('b2', bm[1])
    print('b3', bm[2])
    print('b4', bm[3])
	
    print("All tests passed!")

def makeTestGraph2():

    G = Graph()

    a = G.addVarNode('a', 5)
    b = G.addVarNode('b', 5)
    c = G.addVarNode('c', 5)

    p1 = np.array([[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]] , [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]] , [[3,3,3,3,3],[3,3,3,3,3],[3,3,3,3,3],[3,3,3,3,3],[15,15,15,60,15]] , [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]] , [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]])
    G.addFacNode(p1, a, b, c)

    p2 = np.array([[[1,1,1,1,1],[1,1,1,1,1],[6,12,6,6,6],[1,1,1,1,1],[1,1,1,1,1]] , [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]] , [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]] , [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]] , [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]])
    G.addFacNode(p2, a, b, c)

    p3 = np.array([[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]] , [[2,2,2,2,2],[2,2,2,2,2],[2,2,2,2,2],[8,8,24,8,8],[2,2,2,2,2]] , [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]] , [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]] , [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]])
    G.addFacNode(p3, a, b, c)

    return G

def testTestGraph2():
    """ Automated test case
    """
    G = makeTestGraph2()
    marg = G.marginals()
    brute = G.bruteForce()

    # check the marginals
    am = marg['a']
    print('a1', am[0])
    print('a2', am[1])
    print('a3', am[2])
    print('a4', am[3])
    print('a5', am[4])

    bm = marg['b']
    print('b1', bm[0])
    print('b2', bm[1])
    print('b3', bm[2])
    print('b4', bm[3])
    print('b5', bm[4])

    cm = marg['c']
    print('c1', cm[0])
    print('c2', cm[1])
    print('c3', cm[2])
    print('c4', cm[3])
    print('c5', cm[4])
	
    print("All tests passed!")	
	
# standard run of test cases
testToyGraph()
testTestGraph()
testTestGraph2()