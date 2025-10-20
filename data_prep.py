from langchain_community.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
""" 
Nessa etapa vou deixar o arquivo com minhas informações em um formato vertorizado.
Além de criar chunks, ou seja, bloco de informações cada bloco terá seu vetor, e ao realizar a pergunta o llm poderá avaliar com quais 
chunks a pergunta do usuário está mais relacionada. Dessa forma ganhamos em tempo de processamenrto e custo, porque ao inves de fazer o
llm ler todas as informações, ele irá avaliar somente aquele pedaço do texto que está mais relacionado a pergunta.
""" 

load_dotenv()
caminho_base = "bases/informacoes_externas_alice.txt"

def ajustes_dataset():
    arquivos = lendo_arquivo()
    chunks = dividir_em_chunks(arquivos)

    # embedding - vetoriza o texto dos chunks
    vetoriza_chunks(chunks)

def lendo_arquivo():
    loader = TextLoader(
        caminho_base,
        encoding="utf-8",
    )
    return loader.load()

def dividir_em_chunks(arquivo):
    separador = RecursiveCharacterTextSplitter(
        chunk_size = 450,
        chunk_overlap = 100,
        length_function=len,
        add_start_index=True,
    )
    chunks = separador.split_documents(arquivo)
    return chunks

def vetoriza_chunks(arquivo_chunks):
    db = Chroma.from_documents(
        arquivo_chunks,
        OpenAIEmbeddings(),
        persist_directory = "bases_vetorizadas"
    )
    print("Data prep finalizado! Arquivo ajustado salvo.")

ajustes_dataset()
