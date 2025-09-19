from numpy.random import choice
from scipy import spatial

import matplotlib.pyplot as plt
import random
import math
import pandas as pd
import os
import string

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
    
def carregar_pontos(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)
    pontos = [Ponto(row["x"], row["y"]) for _, row in df.iterrows()]
    return pontos    
  
# Parâmetros do ACO

n = 15        # número de formigas
p = 0.9       # taxa de evaporação
alfa = 0.6    # importância do feromônio
beta = 0.7    # importância da heurística
iteracoes = 20    # quantidade de iterações que irão acontecer até a parada da otimização
start_index = 0    # ponto de início

# Carregar pontos de um arquivo
arquivo = (r"C:\Users\VLN-000159\OneDrive\Área de Trabalho\Unip_ACO\pontos.csv.txt")
pontos = carregar_pontos(arquivo)
qtd_pontos = len(pontos)

# Gerar rótulos (A, B, C, ...)
rotulos = list(string.ascii_uppercase[:qtd_pontos])

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

# Inicialização da colônia
def inicializar_colonia():
  formigas = []

  for _ in range(n):
    formigas.append(Formiga(pontos[start_index]))
  return formigas

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
historico_distancias = []  # <<< NOVO: lista para guardar evolução das distâncias

print(f"---Algoritmo de otimização de formigas---")
print(f"*ACO com {n} formigas por geração*")
print()

# Letras para pontos
labels = string.ascii_uppercase  

for _ in range(iteracoes):
  print(f"Iteração: {_+1}")
  formigas = inicializar_colonia()

  for formiga in formigas:
    movimentar_formiga(formiga, grafo)

    if melhor_rota is None or distancia_rota(melhor_rota) > distancia_rota(formiga.rota):
      melhor_rota = formiga.rota
      distancia_melhor_rota = distancia_rota(formiga.rota)

  atualizar_feromonios(grafo.caminhos)
  
  # salvar a melhor distância da iteração
  historico_distancias.append(distancia_melhor_rota)

  # mostrando a melhor rota a cada iteracao
  if _ < iteracoes - 1:
    for idx, ponto in enumerate(pontos):
        plt.text(ponto.x + 1, ponto.y + 1, rotulos[idx], fontsize=12, color="blue")

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
    for idx, ponto in enumerate(pontos):
      plt.plot(ponto.x, ponto.y, marker='o', color='g')
      plt.text(ponto.x + 1, ponto.y + 1, rotulos[idx], fontsize=12, color="blue")
  
  x = []
  y = []
  

  for ponto in melhor_rota:
      x.append(ponto.x)
      y.append(ponto.y)

  plt.plot(x, y, color='r')

  plt.show()
  print("Melhor rota:", [labels[pontos.index(p)] for p in melhor_rota])
  print("{:.2f}".format(distancia_melhor_rota))

# --- GRÁFICO FINAL DE EVOLUÇÃO ---
plt.figure(figsize=(8,5))
plt.plot(range(1, iteracoes+1), historico_distancias, marker='o', linestyle='-', color='b')
plt.title("Evolução da Melhor Distância por Iteração (ACO)")
plt.xlabel("Iterações")
plt.ylabel("Distância da Melhor Rota")
plt.grid(True)
plt.show()