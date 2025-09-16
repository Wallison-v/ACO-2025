from numpy.random import choice
from scipy import spatial

import matplotlib.pyplot as plt
import random
import math
import pandas as pd
import os

class Formiga:
  def __init__(self, inicio_idx):
    self.ponto_atual = inicio_idx
    self.rota = [inicio_idx]

  def andar(self, ponto):
    self.ponto_atual = ponto
    self.rota.append(ponto)

  def completou_tour(self, pontos):
        return len(self.rota) == pontos

"""## Ponto, Caminho e Grafo

O grafo é a representação de um conjunto de pontos ligados por caminhos. Para este exemplo irei criar as classes "Ponto" (representa uma coordenada no espaço), "Caminho" (ligação entre dois pontos) e "Grafo" (conjunto de caminhos).
"""

class Ponto:
  def __init__(self, x, y):
    self.x = x
    self.y = y



class Caminho:
  def __init__(self, ponto_i, ponto_j):
    self.ponto_i = ponto_i
    self.ponto_j = ponto_j
    self.comprimento = math.sqrt((ponto_i.x - ponto_j.x)**2 + (ponto_i.y - ponto_j.y)**2)
    self.feromonio = 0
    self.formigas_passantes = []

  def contem(self, formiga): #verifica se o ponto a ser visitado ja não esta listado nos pontos visitados(classe formiga, def andar)
    if self.ponto_i == formiga.ponto_atual:
      return self.ponto_j not in formiga.rota
    elif self.ponto_j == formiga.ponto_atual:
      return self.ponto_i not in formiga.rota
    else:
      return False

  def ponto_adjacente(self, ponto):
    if self.ponto_i == ponto:
      return self.ponto_j
    elif self.ponto_j == ponto:
      return self.ponto_i
    else:
      return None

class Grafo:
  def __init__(self, caminhos):
    self.caminhos = caminhos
    self.melhor_rota = []
    self.comprimento_melhor_rota = 0

  def atualizas_melhor_rota(self, melhor_rota):
    self.melhor_rota = melhor_rota
    self.comprimento_melhor_rota = sum([math.sqrt((i.x - j.x)**2 + (i.y - j.y)**2) for [i, j] in melhor_rota])

  def possiveis_caminhos(self, formiga):
    n = qtd_pontos
    if formiga.completou_tour(n):
        start = formiga.rota[0]
        if formiga.ponto_atual != start:
            # Find the path back to the start
            for caminho in self.caminhos:
                if (caminho.ponto_i == formiga.ponto_atual and caminho.ponto_j == start) or \
                   (caminho.ponto_j == formiga.ponto_atual and caminho.ponto_i == start):
                    return [caminho]
        else:
            return []
    else:
      return [caminho for caminho in self.caminhos if caminho.contem(formiga)]
    
# Função para carregar pontos do CSV
def carregar_pontos(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)
    pontos = [Ponto(row["x"], row["y"]) for _, row in df.iterrows()]
    return pontos    
  
# Parâmetros do ACO
# -----------------------------
n = 15        # número de formigas
p = 0.9       # taxa de evaporação
alfa = 0.6    # importância do feromônio
beta = 0.7    # importância da heurística
iteracoes = 3    # quantidade de iterações que irão acontecer até a parada da otimização
start_index = 0    # ponto de início
#qtd_pontos = len(pontos)  quantidade de pontos do grafo que será gerado

# Carregar pontos de um arquivo
arquivo = (r"Unip_ACO\pontos.csv.txt")
pontos = carregar_pontos(arquivo)
qtd_pontos = len(pontos)
   
  
# for _ in range(qtd_pontos):
  # pontos.append(Ponto(random.uniform(-100, 100), random.uniform(-100, 100)))
  # pontos.append(Ponto(-22.30, -1.88))

'''pontos.append(Ponto(-41.38, -11.28))

pontos.append(Ponto(93.75, 2.76))
pontos.append(Ponto(39.17, 19.33))
pontos.append(Ponto(28.89, -12.84))
pontos.append(Ponto(-56.75, 18.91))
pontos.append(Ponto(-31.16, -8.51))
pontos.append(Ponto(-23.15, 15.76))
pontos.append(Ponto(-4.98, 5.28))
pontos.append(Ponto(62.46, -3.10))
pontos.append(Ponto(1.49, -7.51))
pontos.append(Ponto(62.13, 15.94))
pontos.append(Ponto(18.41, -3.48))
pontos.append(Ponto(57.99, -11.01))
pontos.append(Ponto(3.63, 11.11))
pontos.append(Ponto(-3.02, -13.18))
pontos.append(Ponto(94.58, -7.96))
pontos.append(Ponto(60.19, -2.96))
pontos.append(Ponto(-12.66, -14.04))
pontos.append(Ponto(-11.23, 20.91))'''

# for _ in range(qtd_pontos):
#   print(pontos[_])

# criando os caminhos
caminhos = []

i = 0
while i < len(pontos) - 1:
  j = i + 1

  while j < len(pontos):
    caminhos.append(Caminho(pontos[i], pontos[j]))
    j += 1

  i += 1

# criando o grafo
grafo = Grafo(caminhos)

"""
## Grafo criado

for ponto in pontos:
    plt.plot(ponto.x, ponto.y, marker='o', color='r')

x = []
y = []

for caminho in caminhos:
  x_i = caminho.ponto_i.x
  x_j = caminho.ponto_j.x
  y_i = caminho.ponto_i.y
  y_j = caminho.ponto_j.y

  x_texto = (x_i + x_j) / 2
  y_texto = (y_i + y_j) / 2

  plt.text(x_texto, y_texto, "{:.2f}".format(caminho.comprimento))

  x.append(x_i)
  x.append(x_j)
  y.append(y_i)
  y.append(y_j)

plt.plot(x, y, color='c')

plt.show()

gx = x
gy = y
"""
# Inicialização da colônia
def inicializar_colonia():
  formigas = []

  for _ in range(n):
    #print(f"Iteração: {_}")
    #formigas.append(Formiga(random.choice(pontos)))
    formigas.append(Formiga(pontos[start_index]))

  return formigas

"""## Escolha do caminho"""

def escolher_caminho(possiveis_caminhos):
  denominador = sum([(caminho.feromonio)**alfa * (1 / caminho.comprimento)**beta for caminho in possiveis_caminhos])
  distribuicao_probabilidades = None

  if denominador == 0:
    distribuicao_probabilidades = [1 / len(possiveis_caminhos)  for _ in possiveis_caminhos]
  else:
    distribuicao_probabilidades = [((caminho.feromonio)**alfa * (1 / caminho.comprimento)**beta) / denominador for caminho in possiveis_caminhos]

  return choice(possiveis_caminhos, 1, p=distribuicao_probabilidades)[0]

"""## Atualização de feromônio"""

def distancia_rota(rota):
  distancia_rota = 0

  for i in range(0, len(rota) - 1):
    distancia = math.sqrt((rota[i].x - rota[i + 1].x)**2 + (rota[i].y - rota[i + 1].y)**2)
    distancia_rota += distancia

  return distancia_rota

def atualizar_feromonios(caminhos):
  for caminho in caminhos:
    soma_heuristica = sum([1 / distancia_rota(formiga.rota) for formiga in caminho.formigas_passantes])
    caminho.feromonio = (1 - p) * caminho.feromonio + soma_heuristica
    caminho.formigas_passantes = []

"""## Movimentação da formiga

"""

def movimentar_formiga(formiga, grafo):
  while True:
    possiveis_caminhos = grafo.possiveis_caminhos(formiga)

    if possiveis_caminhos == []:
      break

    caminho_escolhido = escolher_caminho(possiveis_caminhos)
    caminho_escolhido.formigas_passantes.append(formiga)
    formiga.andar(caminho_escolhido.ponto_adjacente(formiga.ponto_atual))
    """## Otimização"""

# Execução do ACO
melhor_rota = None
distancia_melhor_rota = 0

print(f"---Algoritmo de otimização de formigas---")
print(f"*ACO com {n} formigas por geração*")
print()


for _ in range(iteracoes):
  i = 0

  print(f"Iteração: {_+1}")
  formigas = inicializar_colonia()

  for formiga in formigas:
    movimentar_formiga(formiga, grafo)

    if melhor_rota is None or distancia_rota(melhor_rota) > distancia_rota(formiga.rota):
      melhor_rota = formiga.rota
      distancia_melhor_rota = distancia_rota(formiga.rota)

  atualizar_feromonios(grafo.caminhos)

  # mostrando a melhor rota a cada iteracao
  if _ < iteracoes - 1:
    for ponto in pontos:
      if i == 0:
        plt.plot(ponto.x, ponto.y, marker='x', color='b')
        i = i+1
      else:
        plt.plot(ponto.x, ponto.y, marker='o', color='g')

    x = []
    y = []

    for caminho in caminhos:
      x_i = caminho.ponto_i.x
      x_j = caminho.ponto_j.x
      y_i = caminho.ponto_i.y
      y_j = caminho.ponto_j.y
      x_texto = (x_i + x_j) / 2
      y_texto = (y_i + y_j) / 2

      plt.text(x_texto, y_texto, "{:.2f}".format(caminho.comprimento))

      x.append(x_i)
      x.append(x_j)
      y.append(y_i)
      y.append(y_j)

    plt.plot(x, y, color='y')
  else:
    for ponto in pontos:
      plt.plot(ponto.x, ponto.y, marker='o', color='g')
  
  x = []
  y = []
  

  for ponto in melhor_rota:
      x.append(ponto.x)
      y.append(ponto.y)

  plt.plot(x, y, color='r')

  plt.show()
  print("{:.2f}".format(distancia_melhor_rota))
