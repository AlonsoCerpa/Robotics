import numpy as np
import random
from sklearn.metrics import mean_squared_error

#
# Modelo de transición:
#
#                                                s'
#                     s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16
# P(s'|s, izq) =  s1 [[1, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s2  [1, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s3  [0, 1, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s4  [0, 0, 1, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s5  [0, 0, 0, 0, 1, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s6  [0, 0, 0, 0, 1, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s7  [0, 0, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#            s    s8  [0, 0, 0, 0, 0, 0, 1, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s9  [0, 0, 0, 0, 0, 0, 0, 0, 1,  0,  0,  0,  0,  0,  0,  0],
#                s10  [0, 0, 0, 0, 0, 0, 0, 0, 1,  0,  0,  0,  0,  0,  0,  0],
#                s11  [0, 0, 0, 0, 0, 0, 0, 0, 0,  1,  0,  0,  0,  0,  0,  0],
#                s12  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  1,  0,  0,  0,  0,  0],
#                s13  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  1,  0,  0,  0],
#                s14  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  1,  0,  0,  0],
#                s15  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  1,  0,  0],
#                s16  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  1,  0]]
#
#
#                                                s'
#                     s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16
# P(s'|s, der) =  s1 [[0, 1, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s2  [0, 0, 1, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s3  [0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s4  [0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s5  [0, 0, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s6  [0, 0, 0, 0, 0, 0, 1, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s7  [0, 0, 0, 0, 0, 0, 0, 1, 0,  0,  0,  0,  0,  0,  0,  0],
#            s    s8  [0, 0, 0, 0, 0, 0, 0, 1, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s9  [0, 0, 0, 0, 0, 0, 0, 0, 0,  1,  0,  0,  0,  0,  0,  0],
#                s10  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  1,  0,  0,  0,  0,  0],
#                s11  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  1,  0,  0,  0,  0],
#                s12  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  1,  0,  0,  0,  0],
#                s13  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  1,  0,  0],
#                s14  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  1,  0],
#                s15  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  1],
#                s16  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  1]]
#
#
#                                                s'
#                     s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16
# P(s'|s, arr) =  s1 [[1, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s2  [0, 1, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s3  [0, 0, 1, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s4  [0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s5  [0, 0, 0, 0, 1, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s6  [0, 0, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s7  [0, 0, 0, 0, 0, 0, 1, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#            s    s8  [0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s9  [0, 0, 0, 0, 1, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                s10  [0, 0, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                s11  [0, 0, 0, 0, 0, 0, 1, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                s12  [0, 0, 0, 0, 0, 0, 0, 1, 0,  0,  0,  0,  0,  0,  0,  0],
#                s13  [0, 0, 0, 0, 0, 0, 0, 0, 1,  0,  0,  0,  0,  0,  0,  0],
#                s14  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  1,  0,  0],
#                s15  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  1,  0],
#                s16  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  1]]
#
#
#                                                s'
#                     s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16
# P(s'|s, aba) =  s1 [[1, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s2  [0, 1, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s3  [0, 0, 1, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s4  [0, 0, 0, 0, 0, 0, 0, 1, 0,  0,  0,  0,  0,  0,  0,  0],
#                 s5  [0, 0, 0, 0, 0, 0, 0, 0, 1,  0,  0,  0,  0,  0,  0,  0],
#                 s6  [0, 0, 0, 0, 0, 0, 0, 0, 0,  1,  0,  0,  0,  0,  0,  0],
#                 s7  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  1,  0,  0,  0,  0,  0],
#            s    s8  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  1,  0,  0,  0,  0],
#                 s9  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  1,  0,  0,  0],
#                s10  [0, 0, 0, 0, 0, 0, 0, 0, 0,  1,  0,  0,  0,  0,  0,  0],
#                s11  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  1,  0,  0,  0,  0,  0],
#                s12  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  1,  0,  0,  0,  0],
#                s13  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  1,  0,  0,  0],
#                s14  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  1,  0,  0],
#                s15  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  1,  0],
#                s16  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  1]]
#
#
# P = [ P(s'|s, izq), P(s'|s, der), P(s'|s, arr), P(s'|s, aba)]
#
#
# Política:
#          
# policy_0 = [random.randint(0, 3), ..., random.randint(0, 3)]
# para todos los estados s 
# donde 0 = izq, 1 = der, 2 = arr, 3 = aba
#
#
# Premios:
#
#                 s1  s2  s3  s4  s5  s6  s7  s8  s9 s10 s11 s12 s13 s14 s15 s16
# rewards = izq [[rs, rs, rs, rs, rs, rr, rs, rs, rs, rs, rs, rs, rs, rs, rs, rs],
#           der  [rs, rs, rs, rs, rs, rs, rs, rs, rs, rs, rg, rs, rs, rs, rs, rs],
#           arr  [rs, rs, rs, rs, rs, rs, rs, rs, rr, rs, rs, rs, rs, rs, rs, rs],
#           aba  [rs, rs, rs, rs, rs, rs, rs, rg, rs, rs, rs, rs, rs, rs, rs, rs]]
#
# En el anterior arreglo, rewards[izq, s6] = rewards[arr, s9] = rr
# y también rewards[aba, s8] = rewards[der, s11] = rg
#
# rs = Premio del cuadrado blanco, rs pertenece a {-1, 0, +1}
# rr = Premio del cuadrado rojo, rr = -5, a menos que se cambie este valor
# rg = Premio del cuadrado verde, rg = +5, a menos que se cambie este valor
#
#
# Factor de descuento:
#
# gamma = 1
#
#
# Valor:
#
#       s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16
# V_0 = [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0]
 

def get_value_and_action(num_states, num_actions, state, rewards, gamma, P, V):
    max_value = None
    max_action = None
    for action in range(num_actions):
        sum = 0
        for state_prime in range(num_states):
            sum += P[action, state, state_prime] * V[state_prime]
        value = rewards[state, action] + gamma * sum
        if max_value == None or max_value < value:
            max_value = value
            max_action = action
    return max_value, max_action

def simulate_episode(state, num_actions, P, end_states):
    list_states = [state]
    action = random.randint(0, num_actions-1)
    aux_array = P[action, state]
    state = np.where(aux_array == 1)[0][0]
    print(action, state)
    list_states.append(state)
    while state not in end_states:
        action = random.randint(0, num_actions-1)
        aux_array = P[action, state]
        state = np.where(aux_array == 1)[0][0]
        print(action, state)
        list_states.append(state)
    return state

def get_value(num_states, state, rewards, gamma, P, V, policy):
    action = policy[state]
    sum = 0
    for state_prime in range(num_states):
        sum += P[action, state, state_prime] * V[state_prime]
    value = rewards[state, action] + gamma * sum
    return value

def main():
    P = np.array([[[1, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [1, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 1, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 1, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  1,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  1,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  1,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  1,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  1,  0]],

                  [[0, 1, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 1, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  1,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  1,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  1,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  1,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  1,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  1]],

                  [[1, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 1, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 1, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  1,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  1,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  1]],

                  [[1, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 1, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 1, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  1,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  1,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  1,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  1,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  1,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  1,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  1,  0,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  1,  0,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  1,  0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  1]]], dtype=float)

    state_red_sqr = 4
    state_green_sqr = 11
    neighbors_red_sqr = [(5, 0), (8, 2)]
    neighbors_green_sqr = [(7, 3), (10, 1)]
    num_states = P.shape[1]
    num_actions = P.shape[0]
    gamma = 1.0

    ################# EXPERIMENTO 1 #################
    experiments_a = [{"rs" : -1.0}, {"rs" : 0.0}, {"rs" : 1.0}]
    list_policies = []

    for experiment in experiments_a:
        V_old = np.zeros(num_states, dtype=float)
        V_new = np.zeros(num_states, dtype=float)
        rewards = np.full((num_states, num_actions), experiment["rs"], dtype=float)
        for state_action in neighbors_red_sqr:
            rewards[state_action[0], state_action[1]] = -5.0
        for state_action in neighbors_green_sqr:
            rewards[state_action[0], state_action[1]] = 5.0
        for action in range(num_actions):
            rewards[state_red_sqr, action] = 0.0
        for action in range(num_actions):
            rewards[state_green_sqr, action] = 0.0
        policy = np.zeros(num_states, dtype=int)
        iterations = 0
        while True:
            for state in range(num_states):
                value, action = get_value_and_action(num_states, num_actions, state, rewards, gamma, P, V_old)
                V_new[state] = value
                policy[state] = action
            if mean_squared_error(V_old, V_new) < 0.001:
                break
            V_old = np.copy(V_new)
            iterations +=1
            if iterations >= 1000:
                print("No convergio")
                break
        list_policies.append(policy)
        print("Iteraciones =", iterations)
        print("Politica =", policy)
        print("Valor =", V_new)
        print()

    ############### EXPERIMENTO 2 ##################
    V_old = np.zeros(num_states, dtype=float)
    V_new = np.zeros(num_states, dtype=float)
    rewards = np.full((num_states, num_actions), 2.0, dtype=float)
    for state_action in neighbors_red_sqr:
        rewards[state_action[0], state_action[1]] = -3.0
    for state_action in neighbors_green_sqr:
        rewards[state_action[0], state_action[1]] = 7.0
    for action in range(num_actions):
        rewards[state_red_sqr, action] = 0.0
    for action in range(num_actions):
        rewards[state_green_sqr, action] = 0.0
    while True:
        for state in range(num_states):
            V_new[state] = get_value(num_states, state, rewards, gamma, P, V_old, list_policies[0])
        print("Valor =", V_new)
        if mean_squared_error(V_old, V_new) < 0.001:
            break
        V_old = np.copy(V_new)
    print()
    
    ################# EXPERIMENTO 3 #################
    V_old = np.zeros(num_states, dtype=float)
    V_new = np.zeros(num_states, dtype=float)
    rewards = np.full((num_states, num_actions), 2, dtype=float)
    for state_action in neighbors_red_sqr:
        rewards[state_action[0], state_action[1]] = -2.0
    for state_action in neighbors_green_sqr:
        rewards[state_action[0], state_action[1]] = 8.0
    for action in range(num_actions):
        rewards[state_red_sqr, action] = 0.0
    for action in range(num_actions):
        rewards[state_green_sqr, action] = 0.0
    policy = np.zeros(num_states, dtype=int)
    iterations = 0
    while True:
        for state in range(num_states):
            value, action = get_value_and_action(num_states, num_actions, state, rewards, gamma, P, V_old)
            V_new[state] = value
            policy[state] = action
        if mean_squared_error(V_old, V_new) < 0.001:
            break
        V_old = np.copy(V_new)
        iterations +=1
        if iterations >= 1000:
            print("No convergio")
            break
    print("Iteraciones =", iterations)
    print("Politica =", policy)
    print("Valor =", V_new)
    print()
            
main()