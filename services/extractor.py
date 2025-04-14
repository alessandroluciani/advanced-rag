# Estrazione personaggi da PDF narrativi (LangChain + LLaMA3)

from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field
from typing import List
import pandas as pd

from config.setup import config
from langchain_ollama.llms import OllamaLLM


def get_llm(model=config.ollama_model, temperature=config.model_temperature, base_url=config.ollama_url):
    return OllamaLLM(model=model,
                     temperature=temperature,
                     base_url=base_url)

# === CONFIG ===
PDF_PATH = "../knowledge/storia_1.pdf"  # Cambia per ogni file
OUTPUT_CSV = "personaggi_estratti.csv"

# === LLM ===
llm = get_llm(model=config.ollama_model2, temperature=config.model_temperature, base_url=config.ollama_url)

# === Schema Output ===
class Personaggio(BaseModel):
    nome: str = Field(..., description="Nome e Cognome del personaggio")
    storie: List[str] = Field(..., description="Lista delle storie (es. 'storia_1')")
    ruolo: str = Field(..., description="Ruolo narrativo del personaggio")

parser = PydanticOutputParser(pydantic_object=Personaggio)

# === Prompt ===
template = """
Dal testo narrativo fornito, estrai UN personaggio e compila queste informazioni:
- Nome e Cognome
- Storie (es. 'storia_1')
- Ruolo narrativo (protagonista, antagonista, vittima, testimone, investigatore, ecc.)

Testo:
{chunk}

{format_instructions}
"""

prompt = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = LLMChain(llm=llm, prompt=prompt)

# === Carica PDF ===
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# === Estrazione ===
risultati = []
for chunk in chunks:
    try:
        raw_output = chain.run(chunk=chunk.page_content)
        personaggio = parser.parse(raw_output)
        risultati.append(personaggio)
    except Exception as e:
        print(f"Errore nel parsing: {e}")

# === Salva CSV ===
df = pd.DataFrame([r.model_dump() for r in risultati])
df.drop_duplicates(subset=["nome", "ruolo"], inplace=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Estrazione completata. Dati salvati in: {OUTPUT_CSV}")
