import requests
import google.generativeai as genai
import traceback

sensibilidade_letra = 3
sensibilidade_espaco = 3

maximo_letras_repetitiveis = 2

def pos_process(sequencia):
    resultado = ""
    anterior = ''
    letra_significante = ''
    is_transicao = True
    count_letra = 0
    count_trans = 0

    for l in sequencia:
        if is_transicao and l == '*':
            anterior = '*'
            count_letra = 0
            count_trans = 0
            continue
        elif is_transicao and l != '*':
            anterior = l
            count_letra = 1
            is_transicao = False
        elif l != '*':
            if anterior == l:
                count_letra += 1
                continue
            else:
                if count_letra > sensibilidade_letra:
                    resultado = resultado + anterior
                count_letra = 1
                anterior = l
        elif l == '*':
            if anterior != '*':
                if count_letra > sensibilidade_letra:
                    resultado = resultado + anterior
                anterior = '*'
                count_trans = 1
                count_letra = 0
                continue
            if anterior == '*':
                count_trans += 1
                if count_trans > sensibilidade_espaco:
                    is_transicao = True
                    count_trans = 0
    if anterior != '*' and count_letra > sensibilidade_letra:
        resultado = resultado + anterior
        
    print("palavra limpa 1: " + resultado) 

    resultado = pos_process_2(resultado)

    return resultado

def pos_process_2(sequencia):
    count_letra = 0
    count_consoante = 0
    ultima_letra = ""
    resultado = ""

    for l in sequencia:

        if l == ultima_letra:
            count_letra += 1
            if count_letra <= maximo_letras_repetitiveis:
                resultado += l
        else:
            resultado += l
            ultima_letra = l
            count_letra = 1
    print("palavra limpa 2: " + resultado)
    return resultado
