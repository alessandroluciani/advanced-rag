from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import chromadb

from langchain.chains import StuffDocumentsChain

from config.setup import config


class Embedder():
    def __init__(self):
        self.documents = None
        self.split_docs = None
        self.embeddings = OllamaEmbeddings(base_url=config.ollama_url,
                                           model=config.ollama_embedder_model)

        self.chromadb_folder = config.chromadb_folder
        self.db_client = self.get_client()

    def get_client(self):
        return chromadb.PersistentClient(path=self.chromadb_folder)

    def load_documents(self):
        loader = DirectoryLoader(config.docs_folder, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        return documents

    def enrich_with_metadata(self, documents):
        enriched_docs = []
        for doc in documents:
            # Estrazione da contenuto testuale, non dai metadati
            lines = doc.page_content.splitlines()
            print("********")
            print(lines[0])
            print("********")
            story_id = lines[0].strip() if len(lines) > 0 else "unknown"
            story_title = lines[1].strip().strip("“”\"") if len(lines) > 1 else "Titolo sconosciuto"

            doc.metadata["story_id"] = story_id
            doc.metadata["story_title"] = story_title
            enriched_docs.append(doc)
        return enriched_docs


    def split_documents_by_character(self, documents, chunk_size=config.chunk_size_200, chunk_overlap=config.chunk_overlap_200):
        enriched_docs = self.enrich_with_metadata(documents)
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        split_docs = text_splitter.split_documents(enriched_docs)
        return split_docs



    def split_documents_by_recursive(self, documents, chunk_size=config.chunk_size_200, chunk_overlap=config.chunk_overlap_200):
        enriched_docs = self.enrich_with_metadata(documents)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        split_docs = text_splitter.split_documents(enriched_docs)
        return split_docs

    def delete_collection(self, collection_name=None):
        collections = self.db_client.list_collections()
        print("Collections founded:", collections)

        if collection_name in collections:
            collection = self.db_client.get_collection(collection_name)
            if collection.count() > 0:
                print("Deleting collection:", collection_name)
                collection.delete(
                    ids=collection.get()['ids'],
                )
                print("Collection deleted:", collection_name)

    def save_embeddings(self, split_docs, collection_name=None):
        self.delete_collection(collection_name=collection_name)
        print("Saving embeddings to a vector store...")
        Chroma.from_documents(documents=split_docs,
                              embedding=self.embeddings,
                              collection_name=collection_name,
                              client=self.db_client,
                              )

    def get_vector_store(self, collection_name=None):
        vector_store = Chroma(client=self.db_client,
                              embedding_function=self.embeddings,
                              collection_name=collection_name)
        return vector_store

    def get_retriever(self, collection_name=None, k_value=None):
        retriever = self.get_vector_store(collection_name=collection_name).as_retriever(
            search_type=config.retriever_search_type,
            search_kwargs={"k": k_value},
        )
        return retriever

    def start_embedding_200(self):
        documents = self.load_documents()
        split_docs = self.split_documents_by_recursive(documents,
                                          chunk_size=config.chunk_size_200,
                                          chunk_overlap=config.chunk_overlap_200)
        self.save_embeddings(split_docs, collection_name=config.knowledge_200_collection)

    def start_embedding_500(self):
        documents = self.load_documents()
        split_docs = self.split_documents_by_recursive(documents,
                                          chunk_size=config.chunk_size_500,
                                          chunk_overlap=config.chunk_overlap_500)
        self.save_embeddings(split_docs, collection_name=config.knowledge_500_collection)

    def start_embedding_1000(self):
        documents = self.load_documents()
        split_docs = self.split_documents_by_recursive(documents,
                                          chunk_size=config.chunk_size_1000,
                                          chunk_overlap=config.chunk_overlap_1000)
        self.save_embeddings(split_docs, collection_name=config.knowledge_1000_collection)
