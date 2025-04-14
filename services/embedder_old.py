
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from config.setup import config

import chromadb
from chromadb.config import Settings


class Embedder():
    def __init__(self):
        self.documents = None
        self.split_docs = None
        self.embeddings = OllamaEmbeddings(base_url=config.ollama_url,
                                           model=config.ollama_embedder_model)
        self.knowledge_collection = config.knowledge_collection
        self.chromadb_folder = config.chromadb_folder
        self.db_client = self.get_client()

    def get_client(self):
        """
        Get the Chroma client.
        """
        return chromadb.PersistentClient(path=self.chromadb_folder)

    def load_documents(self):
        """
        Load documents from a directory.
        """
        loader = DirectoryLoader(config.docs_folder, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        self.documents = documents

    def split_documents(self):
        """
        Split documents into smaller chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=True
        )
        self.split_docs = text_splitter.split_documents(self.documents)


    def delete_collection(self):
        """
        Delete the collection from the vector store.
        """

        collections = self.db_client.list_collections()
        print("Collections founded:", collections)


        if self.knowledge_collection in collections:

            collection = self.db_client.get_collection(self.knowledge_collection)
            collection.delete(
                ids=collection.get()['ids'],
            )
            print("Collection deleted:", self.knowledge_collection)


    def save_embeddings(self):
        """
        Save the embeddings to a vector store.
        """
        self.delete_collection()
        print("Saving embeddings to a vector store...")
        Chroma.from_documents(documents=self.split_docs,
                              embedding=self.embeddings,
                              collection_name=self.knowledge_collection,
                              client=chromadb.PersistentClient(path=self.chromadb_folder),
                              client_settings=Settings(allow_reset=True, anonymized_telemetry=False)
                              )


    def get_vector_store(self):
        vector_store = Chroma(client=chromadb.PersistentClient(path=self.chromadb_folder),
                              embedding_function=self.embeddings,
                              collection_name=self.knowledge_collection)
        return vector_store


    def get_retriever(self):
        """
        Get the vector store for the given text.
        :return:
        retriever: The vector store retriever.
        """
        retriever = self.get_vector_store().as_retriever(
            search_type=config.retriever_search_type,
            search_kwargs={"k": config.retriever_k_value_archive},
        )
        return retriever


    def start_embedding(self):
        """
        Start the embedding process.
        """
        self.load_documents()
        self.split_documents()
        self.save_embeddings()
