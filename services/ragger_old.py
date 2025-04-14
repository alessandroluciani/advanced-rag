
import json

from langchain.chains import SimpleSequentialChain
from langchain.agents import Tool, initialize_agent, AgentType

from langchain.chains.question_answering.map_reduce_prompt import system_template
from langchain_core.messages.tool import tool_call
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM


from langchain_community.llms import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import LLMChain


from langchain.chains import RetrievalQA
from langchain.retrievers.self_query.base import SelfQueryRetriever
from sympy.testing.pytest import tooslow

from services.embedder import Embedder

from config.setup import config



class Ragger():
    def __init__(self):
        self.llm = self.get_llm()
        self.vector_store = Embedder().get_vector_store()
        self.retriever = Embedder().get_retriever()
        self.multiquery_retriever = self.get_multiquery_retriever()


    @staticmethod
    def get_llm():
        return OllamaLLM(model=config.ollama_model,
                         temperature=config.model_temperature,
                         base_url=config.ollama_url)

    def get_multiquery_retriever(self):
        """
        Get the multi-query retriever.
        """
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.retriever,
            llm=self.llm
        )
        return multi_query_retriever


    def ollama_streaming(self, question):
        for chunks in self.generic_agent(question).stream(question):
            yield chunks
        #
        # for chunks in self.llm.stream(question):
        #     yield chunks


    def generic_agent(self, question):
        """
        Create a generic agent for the model.
        """
        template = """
                        ROLE: Rispondi alla 'QUESTION' utilizzando il "CONTEXT.
                        ---
                        QUESTION: {question}
                        ---
                        CONTEXT: {context}
                        """

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        retriever = self.multiquery_retriever

        def format_docs(docs):
            print("Documenti presi in considerazione:", len(docs))
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        return rag_chain
