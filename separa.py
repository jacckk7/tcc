import letters_numbers as ln

dic_breno = {}
dic_pedro = {}
dic_world_breno = {}
dic_world_pedro = {}

for chave, valores in ln.vetor.items():
  vetor_breno = []
  vetor_pedro = []
  
  count = 0
  for valor in valores:
    if count < 10:
      vetor_breno.append(valor)
      count += 1
    else:
      vetor_pedro.append(valor)
      count += 1
      
  dic_breno[chave] = vetor_breno
  dic_pedro[chave] = vetor_pedro
  
for chave, valores in ln.vetor_word.items():
  vetor_breno = []
  vetor_pedro = []
  
  count = 0
  for valor in valores:
    if count < 10:
      vetor_breno.append(valor)
      count += 1
    else:
      vetor_pedro.append(valor)
      count += 1
      
  dic_world_breno[chave] = vetor_breno
  dic_world_pedro[chave] = vetor_pedro
  

nome_arquivo = "vetores_separados.py"

with open(nome_arquivo, "w", encoding="utf-8") as arquivo:
    # Escreve o dicionário como uma variável
    arquivo.write(f"vetor_breno = {repr(dic_breno)}\n")
    arquivo.write(f"vetor_pedro = {repr(dic_pedro)}\n")
    arquivo.write(f"vetor_world_breno = {repr(dic_world_breno)}\n")
    arquivo.write(f"vetor_world_pedro = {repr(dic_world_pedro)}\n")