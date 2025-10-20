from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.manager import get_openai_callback
import time

load_dotenv()
caminho_dados = "bases_vetorizadas"
caminho_instrucoes_txt = "bases/instrucoes_contexto.txt"

with open(caminho_instrucoes_txt, 'r', encoding='utf-8') as arquivo:
    instrucoes_txt = arquivo.read()

proompt_template = """
Responda a pergunta do usuário:
{pergunta}

Levando em consideração os seguintes padrões:
{instrucoes}

Com base nessas informações:
{informacoes_relevantes}

Se você não encontrar a resposta a pergunta do usuário nessas informações,
reponda que não sabe dizer isso.
"""

def perguntando(pergunta_input):
    # pergunta = input("Escreva sua pergunta: ")

    funcao_embedding = OpenAIEmbeddings()
    dados = Chroma(
        persist_directory=caminho_dados, 
        embedding_function=funcao_embedding
    )

    # vamos comparar a pergunta com os dados vetorizados 
    resultados = dados.similarity_search_with_relevance_scores(pergunta_input, k=3)
    # print(resultados)
    # bem legal, dependendo da pergunta o score muda de forma que faz sentido
    # acho que posso diminiur o chunk_size porque as frases ficaram bem longas

    if len(resultados) == 0 or resultados[0][1] < 0.4:
        print("Não foi possível encontrar resposta para essa pergunta através do banco de dados fornecido pela Alice.")
        return None
    
    resultados_relevantes = []
    for resultado in resultados:
        texto = resultado[0].page_content
        resultados_relevantes.append(texto)
    
    informacoes_relevantes = "\n\n ---- \n\n".join(resultados_relevantes)


    prompt = ChatPromptTemplate.from_template(proompt_template)
    modelo = ChatOpenAI(model="gpt-4o")
    parser = StrOutputParser()

    # chain = prompt | modelo | parser
    # resposta = chain.invoke({
    #     "pergunta": pergunta_input,
    #     "instrucoes": instrucoes_txt,
    #     "informacoes_relevantes": informacoes_relevantes,
    # })
    # return resposta
    # nao vou adicionar o parser na chain pq quero ver a quantidade de token

    chain = prompt | modelo

    with get_openai_callback() as cb:
        start = time.perf_counter()
        resultado = chain.invoke({
            "pergunta": pergunta_input,
            "instrucoes": instrucoes_txt,
            "informacoes_relevantes": informacoes_relevantes,
        })
        end = time.perf_counter()
        tempo_total = end - start

        print("Texto final:", parser.invoke(resultado))
        print("Tokens usados:", cb.total_tokens)
        print("Prompt tokens:", cb.prompt_tokens)
        print("Completion tokens:", cb.completion_tokens)
        print("Tempo total de execução:", round(tempo_total, 2), "segundos")

        return parser.invoke(resultado)
