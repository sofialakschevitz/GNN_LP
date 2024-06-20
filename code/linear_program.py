import torch
import numpy as np
import random as rd

from scipy.optimize import linprog
from torch.utils.data import TensorDataset, DataLoader

#Função que gera um problema de programação linear aleatório
def generate_random_lp(num_variables, num_constraints, nnz = 100):
    #Gera coeficientes aleatórios para a função objetivo
    c = np.random.uniform(-1, 1, num_variables) * 0.01

    #Gera a matriz de coeficientes das restrições
    A = np.zeros((num_constraints, num_variables))
    EdgeIndex = np.zeros((nnz, 2))
    EdgeIndex1D = rd.sample(range(num_constraints * num_variables), nnz)
    EdgeFeature = np.random.normal(0, 1, nnz)
    
    for l in range(nnz):
        i = int(EdgeIndex1D[l] / num_variables)
        j = EdgeIndex1D[l] - i * num_variables
        EdgeIndex[l, 0] = i
        EdgeIndex[l, 1] = j
        A[i, j] = EdgeFeature[l]

    #Gera o vetor de termos independentes (do lado direito) das restrições
    b = np.random.uniform(-1, 1, num_constraints)

    #Gera os limites das variáveis de decisão
    bounds = np.random.normal(0, 10, size = (num_variables, 2))
        
    for j in range(num_variables):
        if bounds[j, 0] > bounds[j, 1]:
            temp = bounds[j, 0]
            bounds[j, 0] = bounds[j, 1]
            bounds[j, 1] = temp

    #Gera o tipo de cada restrição (0 para <=, 1 para =)
    constraint_types = np.random.choice([0, 1], size=num_constraints, p=[0.7, 0.3])

    #Garante que pelo menos uma solução viável existe para cada restrição de igualdade individualmente
    for i in range(num_constraints):
        if constraint_types[i] == 1:  # Equality constraint
            b[i] = np.dot(A[i], np.random.rand(num_variables))

    #Retorna o problema de programação linear gerado
    return c, A, b, constraint_types, bounds, EdgeIndex

#Função que resolve um programa linear pela função linprog do scipy
def solve_lp(c, A, b, constraint_types, bounds):
    res = linprog(c, A_ub=A[constraint_types == 0], b_ub=b[constraint_types == 0],
                  A_eq=A[constraint_types == 1], b_eq=b[constraint_types == 1], bounds=bounds)

    #Retorna a solução do problema e se encontrou uma solução factível
    return res.x, res.status
# res.status:
# 0: Otimização bem-sucedida.
# 1: Atingido o número máximo de iterações.
# 2: Problema não possui uma solução factível.
# 3: Problema não possui solução ótima finita (problema ilimitado).
# Outros códigos podem indicar diferentes tipos de falhas.

#Função que gera um conjunto de problemas de programação linear
def loader_lp(num_batches, num_variables, num_constraints, out_func):
    batches_c = [None] * num_batches
    batches_A = [None] * num_batches
    batches_edge_index = [None] * num_batches
    batches_b = [None] * num_batches
    batches_constraint_types = [None] * num_batches
    batches_lower_bounds = [None] * num_batches
    batches_upper_bounds = [None] * num_batches
    batches_solutions = [None] * num_batches
    batches_feasibility = [None] * num_batches

    #Para verificar a viabilidade do problema, os dados gerados contém problemas factíveis e não factíveis
    if out_func == 'feas':
        #Converte as componentes do problema e a solução para tensores do PyTorch e as armazena nas listas correspondentes
        for i in range(num_batches):
            c, A, b, constraint_types, bounds, EdgeIndex = generate_random_lp(num_variables, num_constraints)
            solution, feasibility = solve_lp(c, A, b, constraint_types, bounds)

            lower_bounds, upper_bounds = zip(*bounds)

            batches_c[i] = torch.tensor(c, dtype=torch.float32)
            batches_A[i] = torch.tensor(A, dtype=torch.float32)
            batches_b[i] = torch.tensor(b, dtype=torch.float32)
            batches_constraint_types[i] = torch.tensor(constraint_types, dtype=torch.float32)
            batches_lower_bounds[i] = torch.tensor(lower_bounds, dtype=torch.float32)
            batches_upper_bounds[i] = torch.tensor(upper_bounds, dtype=torch.float32)
            batches_edge_index[i] = torch.tensor(EdgeIndex, dtype=torch.int64)

            if type(solution) != type(None):
                batches_solutions[i] = torch.tensor(solution, dtype=torch.float32)
            else:
                batches_solutions[i] = torch.zeros(num_variables, dtype=torch.float32)

            batches_feasibility[i] = torch.tensor(1 if feasibility != 2 else 0, dtype=torch.float32)

        dataset = TensorDataset(
            torch.stack(batches_c),
            torch.stack(batches_A),
            torch.stack(batches_b),
            torch.stack(batches_constraint_types),
            torch.stack(batches_lower_bounds),
            torch.stack(batches_upper_bounds),
            torch.stack(batches_solutions),
            torch.stack(batches_feasibility),
            torch.stack(batches_edge_index)
        )

        dataloader = DataLoader(dataset, batch_size=num_batches)
        return dataloader
    #Para verificar o valor objetivo e a solução do problema, os dados gerados contém apenas problemas factíveis
    else:
        i = 0
        while i < num_batches:
            c, A, b, constraint_types, bounds, EdgeIndex = generate_random_lp(num_variables, num_constraints)
            solution, feasibility = solve_lp(c, A, b, constraint_types, bounds)

            #Se a solução não é factível, não a armazena
            if type(solution) == type(None):
                continue

            lower_bounds, upper_bounds = zip(*bounds)

            batches_c[i] = torch.tensor(c, dtype=torch.float32)
            batches_A[i] = torch.tensor(A, dtype=torch.float32)
            batches_b[i] = torch.tensor(b, dtype=torch.float32)
            batches_constraint_types[i] = torch.tensor(constraint_types, dtype=torch.float32)
            batches_lower_bounds[i] = torch.tensor(lower_bounds, dtype=torch.float32)
            batches_upper_bounds[i] = torch.tensor(upper_bounds, dtype=torch.float32)
            batches_edge_index[i] = torch.tensor(EdgeIndex, dtype=torch.int64)
            batches_solutions[i] = torch.tensor(solution, dtype=torch.float32)
            batches_feasibility[i] = torch.tensor(1 if feasibility != 2 else 0, dtype=torch.float32)

            i += 1

        dataset = TensorDataset(
            torch.stack(batches_c),
            torch.stack(batches_A),
            torch.stack(batches_b),
            torch.stack(batches_constraint_types),
            torch.stack(batches_lower_bounds),
            torch.stack(batches_upper_bounds),
            torch.stack(batches_solutions),
            torch.stack(batches_feasibility),
            torch.stack(batches_edge_index)
        )

        dataloader = DataLoader(dataset, batch_size=num_batches)
        return dataloader

#Função que gera o conjunto de programas lineares que serão usados no modelo
def gen_data(num_data, batch_size, num_variables, num_constraints, out_func):
    #num_data: número de conjuntos de problemas lineares a serem gerados
    #batch_size: número de problemas lineares em cada conjunto
    data = [None] * num_data
    for i in range(num_data):
        dataloader = loader_lp(batch_size, num_variables, num_constraints, out_func)
        data[i] = dataloader
    return data