
import json

from langchain.chains.sequential import SimpleSequentialChain
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
from langchain.schema import SystemMessage, HumanMessage

from langchain.chains import RetrievalQA
from langchain.retrievers.self_query.base import SelfQueryRetriever
from sympy.testing.pytest import tooslow
from langchain_core.runnables import RunnableLambda


from services.embedder import Embedder

from config.setup import config



class Ragger():
    def __init__(self):
        self.llm = self.get_llm()

        self.retriever_200_5 = Embedder().get_retriever(collection_name=config.knowledge_200_collection,
                                                  k_value=config.retriever_k_5)
        self.retriever_500_5 = Embedder().get_retriever(collection_name=config.knowledge_500_collection,
                                                  k_value=config.retriever_k_5)
        self.retriever_1000_5 = Embedder().get_retriever(collection_name=config.knowledge_1000_collection,
                                                  k_value=config.retriever_k_5)

        self.retriever_200_10 = Embedder().get_retriever(collection_name=config.knowledge_200_collection,
                                                        k_value=config.retriever_k_10)
        self.retriever_500_10 = Embedder().get_retriever(collection_name=config.knowledge_500_collection,
                                                        k_value=config.retriever_k_10)
        self.retriever_1000_10 = Embedder().get_retriever(collection_name=config.knowledge_1000_collection,
                                                         k_value=config.retriever_k_10)


        self.multiquery_retriever_200_5 = self.get_multiquery_retriever(self.retriever_200_5)
        self.multiquery_retriever_500_5 = self.get_multiquery_retriever(self.retriever_500_5)
        self.multiquery_retriever_1000_5 = self.get_multiquery_retriever(self.retriever_1000_5)

        self.multiquery_retriever_200_10 = self.get_multiquery_retriever(self.retriever_200_10)
        self.multiquery_retriever_500_10 = self.get_multiquery_retriever(self.retriever_500_10)
        self.multiquery_retriever_1000_10 = self.get_multiquery_retriever(self.retriever_1000_10)

        self.rag_chain = self.generic_agent()
        self.team_chain = self.team_chain()


    @staticmethod
    def get_llm(model=config.ollama_model, temperature=config.model_temperature, base_url=config.ollama_url):
        return OllamaLLM(model=model,
                         temperature=temperature,
                         base_url=base_url)

    def get_multiquery_retriever(self, retriever=None):
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=self.llm
        )
        return multi_query_retriever


    def ollama_streaming(self, question):

        # ONE AGENT
        for chunks in self.rag_chain.stream(question):
            yield chunks

        # TEAM
        # for chunks in self.team_chain.stream(question):
        #     yield chunks



    def generic_agent(self):
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
        retriever = self.multiquery_retriever_200_5

        def format_docs(docs):
            print("Documenti presi in considerazione:", len(docs))
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.get_llm(model=config.ollama_model,
                               temperature=config.model_temperature,
                               base_url=config.ollama_url)
                | StrOutputParser()
        )

        return rag_chain

    def team_chain(self):

        system_template = """
        Tu sei un investigatore professionista, un esperto di ricerca e analisi delle informazioni.
        Il tuo compito è quello di rispondere a domande complesse utilizzando le informazioni fornite nel CONTEXT.
        Devi sempre dare priorità alla "QUESTION" e utilizzare le informazioni nel CONTEXT per fornire una risposta completa e accurata.
        ---
        """

        template_1 = """
        ROLE: Rispondi alla 'QUESTION' utilizzando il CONTEXT.
        ---
        QUESTION: {question}
        ---
        CONTEXT: {context}
        """

        template_2 = """
        ROLE: Rispondi alla "QUESTION" continuando la "RISPOSTAPRECEDENTE" utilizzando il "CONTEXT".
        ---
        QUESTION: {question}
        ---                 
        RISPOSTAPRECEDENTE: {answer}
        ---       
        CONTEXT: {context}
        """

        template_3 = """        
        ROLE: Rispondi alla 'QUESTION' utilizzando il CONTEXT.
        ---
        QUESTION: {question}
        ---                        
        CONTEXT: {context}
        """

        prompt_1 = PromptTemplate(template=template_1, input_variables=["context", "question"])
        prompt_2 = PromptTemplate(template=template_2, input_variables=["context", "question", "answer"])
        prompt_3 = PromptTemplate(template=template_3, input_variables=["context", "question"])

        def format_docs(docs):
            print("Documenti presi in considerazione:", len(docs))
            return "\n\n".join(doc.page_content for doc in docs)


        agent_1_chain = (
                {
                    "context": self.multiquery_retriever_200_5 | format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt_1
                | self.get_llm(model=config.ollama_model,
                               temperature=config.model_temperature,
                               base_url=config.ollama_url)
                | StrOutputParser()
        )

        agent_2_chain = (
                {
                    "answer": agent_1_chain,
                    "context": self.multiquery_retriever_200_5 | format_docs,
                    "question": RunnablePassthrough(),
                }
                | prompt_2
                | self.get_llm(model=config.ollama_model,
                               temperature=config.model_temperature,
                               base_url=config.ollama_url)
                | StrOutputParser()
        )


        team_chain = (
                agent_2_chain
                | {
                    "context": self.multiquery_retriever_200_5 | format_docs,
                    "question": RunnablePassthrough(),
                }
                | prompt_3
                | self.get_llm(model=config.ollama_model4,
                               temperature=config.model_temperature,
                               base_url=config.ollama_url)
                | StrOutputParser()
        )

        return agent_2_chain




    # Un esempio di tool di esempio (puoi mettere un retriever, un documento analyzer, ecc.)
    def example_tool_func(self, query):
        # usa il retriever per ottenere i documenti
        retriever = self.multiquery_retriever_200_5
        docs = retriever.invoke(query)
        # formatta i documenti come stringa
        formatted_docs = "\n\n".join(doc.page_content for doc in docs)
        return formatted_docs


    def run_iterative_agent_with_dynamic_context(self, retriever_tool, question, iterations=3, verbose=True):

        llm = self.get_llm(model=config.ollama_model,
                               temperature=config.model_temperature,
                               base_url=config.ollama_url)

        # Definisce un template di prompt
        prompt_template = PromptTemplate(
            input_variables=["question", "context", "previous_answer"],
            template=(
                "Devi analizzare delle storie poliziesche inventate da me. Rispondi alla DOMANDA e sfrutta il CONTEXT:\n"
                "\n---\n\n"
                "CONTEXT: \n{context}"
                "\n---\n\n"
                "DOMANDA: \n{question}"
                "\n---\n\n"
                "RISPOSTA PRECEDENTE: \n{previous_answer}"
                "\n---\n\n"
                "Fornisci una risposta raffinata, basata sulla risposta precedente e migliorandola con il contesto fornito."
            )
        )

        # Inizializza l'agente
        agent = initialize_agent(
            tools=[retriever_tool],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
            handle_parsing_errors = True
        )

        previous_answer = "Nessuna risposta iniziale."
        for i in range(iterations):
            if verbose:
                print(f"\n🔁 Iterazione {i + 1}...\n")

            # Recupera il contesto aggiornato usando il retriever
            context = retriever_tool.func(question if i == 0 else previous_answer)

            # Costruisce il prompt per questa iterazione
            full_prompt = prompt_template.format(
                question=question,
                context=context,
                previous_answer=previous_answer
            )

            # Fa girare l'agente con il prompt
            previous_answer = agent.run(full_prompt)
        return previous_answer


    def start_test(self, question):

        retriever_tool = Tool(
            name="RetrieverDB",
            func=self.example_tool_func,
            description="Strumento per ricercare nel db vettoriale per rispondere a una query.",
        )

        response = self.run_iterative_agent_with_dynamic_context(
            retriever_tool=retriever_tool,
            question=question,
            iterations=3
        )


        print("\n🧠 Risposta finale:", response)

