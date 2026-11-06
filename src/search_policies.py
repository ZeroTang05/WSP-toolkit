"""Finite-space controls; constrained EI is a declared simplified BO baseline.

Reference: Gardner et al. ICML 2014, https://proceedings.mlr.press/v32/gardner14.html
Fixed RBF hyperparameters are used here, not their full experimental recipe.
"""
import itertools
import math
import numpy as np


GRID=list(itertools.product([0.,.5,1.,1.5],repeat=4))
FIXED=[(0.,)*4,(1.,)*4,(.5,)*4,(1.5,)*4]+[
    tuple(value if i==axis else 1. for i in range(4)) for axis in range(4) for value in [0.,.5,1.5]]


def available(records):
    used={tuple(row['coordinates']) for row in records}
    return [point for point in GRID if point not in used]


def incumbent(records,guard=True):
    valid=[row for row in records if not guard or row['feasible']]
    if not valid:raise ValueError('Base candidate must remain feasible')
    return min(valid,key=lambda row:(row['domain_risk'],row['index']))


def cdf(x):return np.array([.5*(1+math.erf(float(value)/math.sqrt(2))) for value in np.ravel(x)]).reshape(np.shape(x))


def gp_predict(x,y,query):
    x=np.asarray(x,dtype=float);query=np.asarray(query,dtype=float);y=np.asarray(y,dtype=float)
    scale=max(float(y.std()),1e-6);mean=float(y.mean());z=(y-mean)/scale
    def rbf(a,b):return np.exp(-.5*np.square((a[:,None,:]-b[None,:,:])/.75).sum(-1))
    K=rbf(x,x)+np.eye(len(x))*1e-6;cross=rbf(query,x)
    prediction=mean+scale*cross@np.linalg.solve(K,z)
    variance=np.maximum(1.-(cross*np.linalg.solve(K,cross.T).T).sum(-1),1e-12)*scale**2
    return prediction,np.sqrt(variance)


def propose(policy,records,rng,guard=True):
    if len(records)<2:return (0.,)*4 if not records else (1.,)*4
    remaining=available(records)
    if not remaining:raise ValueError('Search space exhausted')
    if policy=='fixed':return next(point for point in FIXED if point in remaining)
    if policy=='random':return remaining[int(rng.integers(len(remaining)))]
    best=incumbent(records,guard)
    if policy=='greedy':
        neighbors=[point for point in remaining if sum(a!=b for a,b in zip(point,best['coordinates']))==1]
        # Deterministic cyclic coordinate proposals; objective feedback chooses the center.
        return (neighbors or remaining)[0]
    if policy!='bo':raise ValueError('Unknown non-LLM policy')
    x=[row['coordinates'] for row in records]
    mean,std=gp_predict(x,[row['domain_risk'] for row in records],remaining)
    improvement=best['domain_risk']-mean;z=improvement/std
    ei=improvement*cdf(z)+std*np.exp(-.5*z*z)/math.sqrt(2*math.pi)
    if guard:
        margin,sigma=gp_predict(x,[row['guard_upper95']-row['guard_margin'] for row in records],remaining)
        ei*=cdf(-margin/sigma)
    return remaining[int(np.argmax(ei))]
