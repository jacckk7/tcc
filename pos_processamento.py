import requests
import ollama
import google.generativeai as genai
import traceback

sensibilidade_letra = 3
sensibilidade_espaco = 3

def corrigir_palavra(palavra):
    url = "https://api.languagetool.org/v2/check"
    data = {
        'text': palavra,
        'language': 'pt-BR'
    }

    response = requests.post(url, data=data)
    result = response.json()

    # Verifica se há alguma sugestão de correção
    matches = result.get("matches", [])
    if matches:
        # Pega a primeira sugestão da primeira correção
        sugestoes = matches[0].get("replacements", [])
        if sugestoes:
            return sugestoes[0]["value"]  # Palavra mais provável

    return palavra  # Retorna a original se não houver correções

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
        
    pergunta = "Encontre a palavra no português e que seja nome de pessoa, cidade estado ou país que melhor se aproxima da palavra " + resultado + ", lembrando que existem letras erradas e repetidas dentro dela. Me dê apenas a resposta final."
    print("palavra limpa: " + resultado)
    print("palavra corrigida: " + corrigir_palavra(resultado))
    """ response = ollama.chat(
        model='llama3',
        messages=[
            {"role": "user", "content": pergunta}
        ]
    )

    print(response['message']['content'])
    return "palavra limpa: " + resultado + "\n" + "palavra corrigida: " + corrigir_palavra(resultado) + "\n" + response['message']['content'] + "\n\n" """
    
    try:
        genai.configure(api_key="AIzaSyAiDaoJg5rIaxQqNfA51kGrQW_7_59V77A")
    
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(pergunta)
        print(response.text)
    except Exception as e:
        print("Erro detectado:", type(e).__name__, "-", e)

    return "palavra limpa: " + resultado + "\n" + "palavra corrigida: " + corrigir_palavra(resultado) + "\n" + response.text + "\n"
    