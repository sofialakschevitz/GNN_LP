import torch
import numpy as np
import torch.nn as nn

from torch_geometric.nn import MLP
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import MessagePassing

#Classe que representa dados em grafos bipartidos ponderados
class BipartiteData(Data):
    #Todos os parâmetros são torch.tensors
    def __init__(self, x_s, x_t, edge_index, edge_weight, num_nodes):
        super().__init__(num_nodes=num_nodes)
        self.x_s = x_s #Features dos nós do lado esquerdo
        self.x_t = x_t #Features dos nós do lado direito
        self.edge_index = edge_index   #Índices das arestas
        self.edge_weight = edge_weight #Pesos das arestas
        
    #Especifica como o tamanho do grafo deve ser aumentado quando novos dados são adicionados
    #Garante a atualização adequada do grafo
    def __inc__(self, key, value, *args, **kwargs):
        #key: atributo a ser incrementado ao grafo (ex: edge_index, edge_weight, x_s, x_t)
        #value : valor do atributo (ex: torch.tensor)
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)

#Camada de convolução para grafos
class CamadaGCN(MessagePassing):
    #Inicializa dois perceptrons multicamadas (MLP), com camadas ocultas.
    #Esses MLPs são usados nas etapas de mensagens e atualização da camada GCN.
    def __init__(self, in_channels, hidden_channels, out_channels):
        #in_channels: features de entrada dos nós
        #hidden_channels: número de camadas ocultas
        #out_channels: features de saída dos nós após a convolução
        super(CamadaGCN, self).__init__(aggr=None)
        self.mlp1 = MLP([in_channels + out_channels, hidden_channels, out_channels])
        self.mlp2 = MLP([in_channels, hidden_channels, out_channels])

    #Função que define como os dados são propagados na camada GCN
    def forward(self, x, edge_index, edge_weight, size):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
    
    #Função que define as mensagens que são enviadas entre os nós
    def message(self, x, edge_weight):
        reshaped = edge_weight[:,:,None].reshape((edge_weight.shape[0] * edge_weight.shape[1]) // self.mlp2(x[1]).shape[0],
                                                 self.mlp2(x[1]).shape[0],
                                                 1)
        a = torch.mul(reshaped, self.mlp2(x[1]))
        a = a.reshape(x[0].shape[0], (a.shape[0] * a.shape[1]) // x[0].shape[0], a.shape[2])
        return a
    
    #Função que agrega as mensagens recebidas dos nós vizinhos para calcular as novas características dos nós do grafo
    def aggregate(self, inputs):
        #inputs: representa as mensagens recebidas pelos nós vizinhos durante a propagação
        return torch.sum(inputs, dim=1)

    #Função que calcula as novas características dos nós do grafo utilizando as mensagens agregadas.
    def update(self, aggr_out, x):
        #aggr_out: mensagens agregadas
        #x: características originais dos nós
        return self.mlp1(torch.cat((x[0], aggr_out), dim=1))

class LPGCN(nn.Module):
    #Modelo de GNN para resolver programas lineares
    #O LP é modelado como um grafo bipartido ponderado com restrições no lado esquerdo e variáveis de decisão no lado direito
    #Os problemas de otimização são da forma: min c^T x s.t. Ax (restrições) b, l <= x <= u

    def __init__(self, num_constraints, num_variables, num_layers=5):
        super().__init__()

        self.num_constraints = num_constraints
        self.num_variables = num_variables
        self.num_layers = num_layers

        #Gera números aleatórios inteiros para as dimensões das camadas ocultas
        #As dimensões das camadas ocultas são potências de 2, variando de 2 a 512
        ints = np.random.randint(1, 10, size=self.num_layers)
        dims = [2 ** i for i in ints]
        
        #Codifica as características de entrada no espaço de incorporação
        self.fv_in = MLP([2, 32, dims[0]])
        self.fw_in = MLP([3, 32, dims[0]])

        # Hidden states (convolução)
        self.cv = nn.ModuleList([CamadaGCN(dims[l-1], 32, dims[l]) for l in range(1, self.num_layers)])
        self.cw = nn.ModuleList([CamadaGCN(dims[l-1], 32, dims[l]) for l in range(1, self.num_layers)])
        
        #Função de saída para viabilidade e valor objetivo
        self.f_out = MLP([2 * dims[self.num_layers-1], 32, 1])

        #Função de saída para a solução
        self.fw_out = MLP([3 * dims[self.num_layers-1], 32, 1])

    #Função que constrói o grafo
    def construct_graph(self, c, A, b, constraints, l, u):
        #Features das restrições
        hv = torch.cat((b.unsqueeze(2), constraints.unsqueeze(2)), dim=2)

        #Features das variáveis de decisão
        hw = torch.cat((c.unsqueeze(2), l.unsqueeze(2), u.unsqueeze(2)), dim=2)

        #Arestas
        E = A

        return hv, hw, E

    #Função que inicializa as características (features) dos nós (camada 0)
    def init_features(self, hv, hw):
        #Aplica MLP a cada linha das features das restrições
        hv_0 = []
        for i in range(self.num_constraints):
            hv_0.append(self.fv_in(hv[:, i]))

        #Aplica MLP a cada linha das features das variáveis de decisão
        hw_0 = []
        for j in range(self.num_variables):
            hw_0.append(self.fw_in(hw[:, j]))

        hv = torch.stack(hv_0, dim=1)
        hw = torch.stack(hw_0, dim=1)

        return hv, hw

    #Função que executa as convoluções da esquerda para a direita e da direita para a esquerda para cada camada
    def convs(self, hv, hw, edge_index, E, layer, batch_size):
        hv_l = self.cv[layer]((hv, hw),
                                edge_index,
                                E,
                                (self.num_constraints * batch_size, self.num_variables))
        
        hw_l = self.cw[layer]((hw, hv),
                                torch.flip(edge_index, dims=[1,0]),
                                E.T,
                                (self.num_variables, self.num_constraints * batch_size))

        return hv_l, hw_l
    
    #Função de saída para viabilidade e valor objetivo
    def single_output(self, hv, hw):
        y_out = self.f_out(torch.cat((torch.sum(hv, 1), torch.sum(hw, 1)), dim=1))
        #Se a saída for para viabilidade, retorna um vetor binário indicando se os LPs são viáveis ou não
        #Se a saída for para objetivo, retorna o valor da função objetivo dos LPs
        return y_out
    
    #Função de saída para a solução que retorna a solução aproximada dos LPs
    def sol_output(self, hv, hw):
        sol = []
        for j in range(self.num_variables):
            joint = torch.cat((torch.sum(hv, 1), torch.sum(hw, 1), hw[:, j]), dim=1)
            sol.append(self.fw_out(joint))

        sol = torch.stack(sol, dim=1)

        return sol[:, :, 0]

    #Função que executa o forward pass do modelo
    def forward(self, c, A, b, constraints, l, u, edge_index, phi): #phi = 'feas'
        #phi: tipo de função (feas, obj ou sol)
        #Se a saída for para viabilidade, retorna um vetor binário indicando se os LPs são viáveis ou não
        #Se a saída for para objetivo, retorna o valor da função objetivo dos LPs
        #Se a saída for para solução, retorna a solução aproximada dos LPs

        hv, hw, E = self.construct_graph(c, A, b, constraints, l, u)
        hv, hw = self.init_features(hv, hw)

        batch_size = hv.shape[0]

        graphs = [BipartiteData(x_s=hv[i], x_t=hw[i], edge_index=edge_index[i].T, edge_weight=E[i], num_nodes=self.num_variables+self.num_constraints) for i in range(hv.shape[0])]
        loader = DataLoader(graphs, batch_size=batch_size)
        batch = next(iter(loader))

        hv = batch.x_s
        hw = batch.x_t
        edge_index = batch.edge_index
        E = batch.edge_weight

        #Iteração sobre as camadas
        for l in range(self.num_layers-1):
            hv, hw = self.convs(hv, hw, edge_index, E, l, batch_size)

        hv = hv.reshape(batch_size, hv.shape[0] // batch_size, hv.shape[1])
        hw = hw.reshape(batch_size, hw.shape[0] // batch_size, hw.shape[1])

        if phi == 'feas':
            output = self.single_output(hv,hw)
            bins = [1 if elem >= 1/2 else 0 for elem in output]
            return torch.tensor(bins, dtype=torch.float32, requires_grad=True)
        
        elif phi == 'obj':
            return self.single_output(hv,hw)
        
        elif phi == 'sol':
            return self.sol_output(hv,hw)
        
        else:
            return "Please, choose one type of function: feas, obj or sol"

