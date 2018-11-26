from __future__ import print_function
import json
import numpy as np
import sys

def forward(pi, A, B, O):
  """
  Forward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)
  """
  S = len(pi)
  N = len(O)
  alpha = np.zeros([S, N])
  #print(alpha)
  #sth = np.zeros([N,1])
  ###################################################
  # Q3.1 Edit here
  ###################################################
  for t in range(N):
    k = O[t]
    if t == 0:
      #sth[t] = np.multiply(pi,B[k][:])
      for j in range(S):
        #alpha[j][t] = (pi[j] * B[k][j]) / sth[t]
        #alpha[j][t] = pi[j] * B[j][k]        
        alpha[j, t] = pi[j] * B[j, k]
    else:
      for j in range(S):
        #sth_t = np.zeros([S, 1])
        for i in range(S):
          #sth_t[j] += B[k][j] * A[i][j] * alpha[t - 1][i]
          alpha[j, t] += B[j, k] * A[i, j] * alpha[i, t- 1]
          #alpha[j][t] += B[j][k] * A[i][j] * alpha[i][t - 1]
      #sth[t] = np.sum(sth_t)
      #alpha[:][t] = sth_t / sth[t] 
  #print(alpha)  
  return alpha


def backward(pi, A, B, O):
  """
  Backward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  """
  S = len(pi)
  N = len(O)
  beta = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  T = N - 1
  for t in range(T,-1,-1):
    k = O[t]
    if t == T:
      for i in range(S):
        beta[i][t] = 1
    else:
      for i in range(S):
        beta_it = 0
        for j in range(S):
          beta_it += beta[j][t+1] * A[i][j] * B[j][k]
        beta[i][t] = beta_it
  return beta

def seqprob_forward(alpha):
  """
  Total probability of observing the whole sequence using the forward algorithm

  Inputs:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  T = len(alpha[0])
  #print(T)
  alpha_T = alpha[:, T - 1]
  #print(alpha)
  #print(alpha_T)
  prob = np.sum(alpha_T)
  #print(prob) 
  return prob


def seqprob_backward(beta, pi, B, O):
  """
  Total probability of observing the whole sequence using the backward algorithm

  Inputs:
  - beta: A numpy array beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence
      (in terms of the observation index, not the actual symbol)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  ###################################################
  #print(O)
  k1 = O[0]
  S = beta.shape[0]
  for i in range(S):
    #print(B)
    #print(B[i][k1])
    prob += beta[i, 0] * pi[i] * B[i, k1]

  return prob

def viterbi(pi, A, B, O):
  """
  Viterbi algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - path: A list of the most likely hidden state path k* (in terms of the state index)
    argmax_k P(s_k1:s_kT | x_1:x_T)
  """
  path = []
  ###################################################
  # Q3.3 Edit here
  ###################################################
  S = len(pi)
  N = len(O)
  delta = np.zeros([S, N])
  phi = np.zeros([S, N])
  for t in range(N):
    k = O[t]
    if t == 0:
      for j in range(S):
        delta[j][t] = pi[j] * B[j][k]
    else:
      for j in range(S):
        delt_j = np.zeros([S, 1])
        for i in range(S):
          #print(delta[i][t -1] * A[i][j] * B[j][k])
          delt_j[i] = delta[i][t -1] * A[i][j] * B[j][k]
        #print(np.max(delt_j))      
        delta[j][t] = np.max(delt_j)
        phi[j][t] = np.argmax(delt_j)   
  for t in range(N):
    #print(delta)
    delta_t = delta[:,t]
    #print(delta_t)
    phi_t = phi[:,t]        
    s_idx = np.argmax(delta_t)
    path.append(int(phi_t[s_idx]))
  return path


##### DO NOT MODIFY ANYTHING BELOW THIS ###################
def main():
  model_file = sys.argv[1]
  Osymbols = sys.argv[2]

  #### load data ####
  with open(model_file, 'r') as f:
    data = json.load(f)
  A = np.array(data['A'])
  B = np.array(data['B'])
  pi = np.array(data['pi'])
  #### observation symbols #####
  obs_symbols = data['observations']
  #### state symbols #####
  states_symbols = data['states']

  N = len(Osymbols)
  O = [obs_symbols[j] for j in Osymbols]

  alpha = forward(pi, A, B, O)
  beta = backward(pi, A, B, O)

  prob1 = seqprob_forward(alpha)
  prob2 = seqprob_backward(beta, pi, B, O)
  print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

  viterbi_path = viterbi(pi, A, B, O)

  print('Viterbi best path is ')
  for j in viterbi_path:
    print(states_symbols[j], end=' ')

if __name__ == "__main__":
  main()
