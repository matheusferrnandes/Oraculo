import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate  
from langchain_core.runnables import RunnablePassthrough
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from functools import lru_cache
import logging
from typing import Optional, Dict, List, Tuple

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("API Key da OpenAI não encontrada no arquivo .env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

BASE_DIR = Path("Docs")
if not BASE_DIR.exists():
    raise FileNotFoundError(f"Diretório base não encontrado: {BASE_DIR}")

ESTADOS_BR = {
    'AC': 'Acre', 'AL': 'Alagoas', 'AP': 'Amapá', 'AM': 'Amazonas',
    'BA': 'Bahia', 'CE': 'Ceará', 'DF': 'Distrito Federal', 'ES': 'Espírito Santo',
    'GO': 'Goiás', 'MA': 'Maranhão', 'MT': 'Mato Grosso', 'MS': 'Mato Grosso do Sul',
    'MG': 'Minas Gerais', 'PA': 'Pará', 'PB': 'Paraíba', 'PR': 'Paraná',
    'PE': 'Pernambuco', 'PI': 'Piauí', 'RJ': 'Rio de Janeiro', 'RN': 'Rio Grande do Norte',
    'RS': 'Rio Grande do Sul', 'RO': 'Rondônia', 'RR': 'Roraima', 'SC': 'Santa Catarina',
    'SP': 'São Paulo', 'SE': 'Sergipe', 'TO': 'Tocantins'
}

def expandir_siglas(texto: str) -> str:
    for sigla, nome in ESTADOS_BR.items():
        texto = re.sub(rf'\b{sigla}\b', nome, texto, flags=re.IGNORECASE)
    return texto

def normalizar_nomes(texto: str) -> str:
    texto = expandir_siglas(texto)
    substituicoes = {
        'alagoinhas': 'Alagoinhas (BA)',
        'gov': 'governo',
        'pref': 'prefeitura',
        'tj': 'tribunal de justiça'
    }
    for termo, substituicao in substituicoes.items():
        texto = re.sub(rf'\b{termo}\b', substituicao, texto, flags=re.IGNORECASE)
    return texto

contextos_disponiveis = [
    (categoria.name, subdir.name, subdir)
    for categoria in BASE_DIR.iterdir() if categoria.is_dir()
    for subdir in categoria.iterdir() if subdir.is_dir()
]

categorias_unicas = {cat for cat, _, _ in contextos_disponiveis}
subcategorias_por_categoria = {
    cat: [sub for c, sub, _ in contextos_disponiveis if c == cat]
    for cat in categorias_unicas
}

logger.info("Estrutura de documentos detectada:")
for categoria, subcategorias in subcategorias_por_categoria.items():
    logger.info(f"  {categoria}: {', '.join(subcategorias)}")

contexto_embeddings = {
    (categoria, subcategoria): embeddings.embed_query(normalizar_nomes(f"{categoria} {subcategoria}"))
    for categoria, subcategoria, _ in contextos_disponiveis
}

def preprocessar_pergunta(pergunta: str) -> str:
    pergunta = normalizar_nomes(pergunta.lower())
    
    term_map = {
        'governo': 'GOVERNOS',
        'estado': 'GOVERNOS',
        'prefeitura': 'PREFEITURAS',
        'município': 'PREFEITURAS',
        'tribunal': 'TRIBUNAIS',
        'justiça': 'TRIBUNAIS'
    }
    
    for term, categoria in term_map.items():
        if term in pergunta:
            return f"{categoria} {pergunta}"
    
    return pergunta

@lru_cache(maxsize=100)
def detectar_contexto(pergunta: str, threshold=0.5) -> Optional[Tuple[str, str, Path]]:
    pergunta_processada = preprocessar_pergunta(pergunta)
    pergunta_emb = embeddings.embed_query(pergunta_processada)
    max_sim = -1
    contexto_escolhido = None
    
    for (categoria, subcategoria), contexto_emb in contexto_embeddings.items():
        if re.search(rf'\b{subcategoria.lower()}\b', pergunta_processada):
            logger.info(f"Correspondência exata encontrada para {subcategoria}")
            return next((cat, sub, p) for cat, sub, p in contextos_disponiveis 
                       if cat == categoria and sub == subcategoria)
        
        peso = 1.0
        subcategoria_normalizada = normalizar_nomes(subcategoria.lower())
        
        if any(estado in pergunta_processada for estado in ESTADOS_BR.values()):
            if any(estado in subcategoria_normalizada for estado in ESTADOS_BR.values()):
                peso = 2.0  
        
        sim = cosine_similarity([pergunta_emb], [contexto_emb])[0][0] * peso
        
        if sim > max_sim:
            max_sim = sim
            contexto_escolhido = (categoria, subcategoria)
    
    logger.info(f"Similaridade máxima: {max_sim:.2f} para {contexto_escolhido}")
    
    if max_sim < threshold:
        return None
    
    return next((cat, sub, p) for cat, sub, p in contextos_disponiveis 
               if cat == contexto_escolhido[0] and sub == contexto_escolhido[1])

def carregar_documentos_contexto(categoria: str, subcategoria: str, path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(f"Diretório do contexto não encontrado: {path}")
    
    documentos = []
    for pdf_path in path.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            documentos.extend(docs)
            logger.info(f"Carregado: {pdf_path.name}")
        except Exception as e:
            logger.error(f"Erro ao carregar {pdf_path}: {str(e)}")
    
    if not documentos:
        logger.warning(f"Nenhum documento válido encontrado em {path}")
    
    return documentos

vectorstore_cache: Dict[Tuple[str, str], FAISS] = {}

def get_vectorstore(categoria: str, subcategoria: str, path: Path) -> FAISS:
    cache_key = (categoria, subcategoria)
    
    if cache_key in vectorstore_cache:
        logger.info(f"Recuperando vectorstore do cache para {cache_key}")
        return vectorstore_cache[cache_key]
    
    documentos = carregar_documentos_contexto(categoria, subcategoria, path)
    if not documentos:
        raise ValueError(f"Nenhum documento encontrado para {categoria}/{subcategoria}")
    
    vectorstore = FAISS.from_documents(documentos, embeddings)
    vectorstore_cache[cache_key] = vectorstore
    return vectorstore

prompt_template = """
Você é um assistente especializado em crédito consignado para servidores públicos.
Contexto atual: {categoria} > {subcategoria}
Baseado nos documentos abaixo, responda:

Documentos relevantes:
{context}

Pergunta: {question}
Responda de forma clara e específica ao contexto:"""

prompt = ChatPromptTemplate.from_template(prompt_template)

def oraculo_konsi():
    print("\n" + "="*50)
    print(" ORÁCULO KONSI - CRÉDITO CONSIGNADO")
    print("="*50)
    print("Dica: Para melhores resultados, inclua:")
    print("- Sigla do estado (AL, BA, SP) ou nome completo")
    print("- Tipo de instituição (Governo, Prefeitura, Tribunal)")
    print("\nDigite sua pergunta ou 'sair' para encerrar.\n")
    
    while True:
        pergunta = input("\nSua pergunta: ").strip()
        if pergunta.lower() in ["sair"]:
            print("Encerrando o Oráculo. Até mais!")
            break
        
        if len(pergunta) < 5:
            print("Por favor, faça uma pergunta mais detalhada.")
            continue
        
        contexto = detectar_contexto(pergunta)
        if not contexto:
            print("Não foi possível identificar um contexto relevante.")
            print("Sugestões:")
            print("- Para Alagoas: 'GOV AL' ou 'Governo Alagoas'")
            print("- Para Bahia: 'GOV BA' ou 'Prefeitura Salvador'")
            continue
        
        categoria, subcategoria, path = contexto
        print(f"\n Contexto identificado: {categoria} > {subcategoria}")
        
        try:
            vectorstore = get_vectorstore(categoria, subcategoria, path)
            retriever = vectorstore.as_retriever()
            
            chain = (
                {
                    "context": retriever,
                    "categoria": lambda x: categoria,
                    "subcategoria": lambda x: subcategoria,
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
            )
            
            resposta = chain.invoke(pergunta)
            print("\n Resposta:")
            print("-"*50)
            print(resposta.content)
            print("-"*50)
            
        except Exception as e:
            logger.error(f"Erro ao processar: {str(e)}")
            print("Ocorreu um erro. Por favor, reformule sua pergunta.")

if __name__ == "__main__":
    oraculo_konsi()