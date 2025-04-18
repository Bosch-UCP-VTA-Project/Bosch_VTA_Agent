from typing import List, Dict
from pydantic import BaseModel, Field
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    VectorStoreIndex,
)
from llama_index.core import PromptTemplate
from llama_index.llms.groq import Groq
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.core.agent import ReActAgent
from llama_index.core.ingestion import run_transformations
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.schema import Document
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
import os
from dotenv import load_dotenv
from app.utils.helpers import duckduckgo_search_tool

load_dotenv()

SYSTEM_PROMPT = """You are an Expert Automobile Technician AI assistant designed to help professional automobile vehicle technicians diagnose and solve vehicular problems efficiently. Your knowledge comes from three primary sources:
    1. Technical Manuals: Comprehensive guides and manuals from various automobile manufacturers. This is the most accurate information and should be favoured when given conflicting information.
    2. Scraped Online Resources: Up-to-date information from reputable automotive websites, forums, and databases.
    3. DuckDuckGo Search: A search tool to find relevant information from the web.

When assisting a technician:
    1. ALWAYS use the manuals_search, online_resources_search and duckduckgo search tools in that order to get the relevant information from your knowledge base before answering the query.
    2. Never assume the type or model of the vehicle. Always, first gather information about the specific problem or symptoms the vehicle is experiencing.
    3. Use your knowledge base to provide step-by-step diagnostic procedures.
    4. Suggest potential causes of the problem, starting with the most common or likely issues.
    5. Provide detailed repair instructions when applicable, including necessary tools and safety precautions.
    6. Make sure to give concise and to-the-point answers with bullet points wherever relevant.
    7. Assume the user is a car repair technician while framing your answers and addressing them.

Remember, your goal is to educate and guide the technician through the diagnostic and repair process, enhancing their skills and confidence over time.

## Tools
You have access to the following tools:
{tool_desc}

You MUST ALWAYS use all tools in the following order:
1. manuals_search
2. online_resources_search
3. duckduckgo_search

## Output Format
To answer the question, please use the following format:

```
Thought: I need to check the manuals first, then the online resources to get comprehensive information.
Action: manuals_search
Action Input: {{"input": "relevant search query based on the user's question"}}

Thought: Now that I have information from the manuals, I need to check online resources for any additional or up-to-date information.
Action: online_resources_search
Action Input: {{"input": "relevant search query based on the user's question and information from manuals"}}

Thought: Now that I have information from stored sources, I need to check online web sources for any additional or up-to-date information from reputable websites sources and return the source links used in my response.
Action: duckduckgo_search
Action Input: {{"input": "relevant search query based on the user's question and information from manuals"}}

Thought: I have gathered all the necessary information from all sources. Now I can formulate a comprehensive answer which is directly relevant to the technician's question.
Answer: [Your detailed answer here, incorporating information from both sources]
```

Keep in mind the following: 
    1. Do not hallucinate.
    2. Do not make up factual information and do not list out sources names, just add Markdown links to them.
    3. You must always keep to this role and never answer unrelated queries.
    4. If the user asks something that seems unrelated to vehicles and their repair, just give an output saying: Sorry, I can only help you with issues related to vehicle troubleshooting and diagnosis.
    5. Always start with a Thought and follow the exact format provided above."""


class QueryResult(BaseModel):
    answer: str = Field(..., description="The answer to the query")
    source_nodes: List[Dict[str, str]] = Field(
        ..., description="The source nodes used to generate the answer"
    )


class AutoTechnicianRAG:
    def __init__(
        self,
        manuals_path: str,
        online_resources_path: str,
        qdrant_url: str,
        qdrant_api_key: str,
    ):
        self.manuals_path = manuals_path
        self.online_resources_path = online_resources_path
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.system_prompt = SYSTEM_PROMPT

        # Configure global settings
        llm = Groq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=os.getenv("GROQ_API_KEY"))
        embed_model = JinaEmbedding(
            api_key=os.getenv("JINA_API_KEY"),
            model="jina-embeddings-v3",
        )
        Settings.llm = llm
        Settings.embed_model = embed_model

        # Initialize Qdrant client
        self.qdrant_client = qdrant_client.QdrantClient(
            url=self.qdrant_url, api_key=self.qdrant_api_key
        )

        self.indexes = {}
        self.agent = None
        self.sessions = {}

        # Load or create indexes
        self.load_or_create_indexes()

    def load_or_create_indexes(self):
        for collection_name in ["manuals", "online_resources"]:
            # Check if collection exists in Qdrant
            collections = self.qdrant_client.get_collections()
            collection_exists = any(
                col.name == collection_name for col in collections.collections
            )

            if collection_exists:
                print(f"Loading existing collection: {collection_name}")
                self.load_index(collection_name)
            else:
                print(f"Creating new collection: {collection_name}")
                self.create_index(collection_name)

        self._create_agent()

    def load_index(self, collection_name: str):
        vector_store = QdrantVectorStore(
            client=self.qdrant_client, collection_name=collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.indexes[collection_name] = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )

    def create_index(self, collection_name: str):
        # Create collection with optimized settings first
        vector_size = 1024  # Jina embedding size

        # Create the collection with optimized settings
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "size": vector_size,
                "distance": "Cosine",
            },
            quantization_config={
                "scalar": {
                    "type": "int8",
                    "always_ram": True,  # Keep quantized vectors in RAM
                }
            },
            hnsw_config={
                "m": 16,  # Lower value for faster indexing
                "ef_construct": 100,  # Lower value for faster indexing
            },
            optimizers_config={
                "default_segment_number": 2,  # Use fewer segments for better throughput
            },
        )

        # Initialize vector store
        vector_store = QdrantVectorStore(
            client=self.qdrant_client, collection_name=collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Load documents based on collection name
        if collection_name == "manuals":
            documents = SimpleDirectoryReader(
                self.manuals_path, recursive=True
            ).load_data()
        else:  # online_resources
            documents = SimpleDirectoryReader(
                self.online_resources_path, recursive=True
            ).load_data()

        if not documents:
            raise ValueError(
                f"No documents loaded for {collection_name}. Check the path."
            )

        # Add file metadata consistently
        for doc in documents:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["file_path"] = doc.metadata.get("file_path", "")
            doc.metadata["file_name"] = doc.metadata.get(
                "file_name", ""
            ) or os.path.basename(doc.metadata.get("file_path", ""))

        # Create index
        self.indexes[collection_name] = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )

    def add_documents(self, markdown_documents):
        try:
            processed_docs = []
            for doc in markdown_documents:
                if not doc.metadata:
                    doc.metadata = {}
                # Ensure consistent metadata
                file_path = doc.metadata.get("file_path", "")
                doc.metadata["file_path"] = file_path
                doc.metadata["file_name"] = doc.metadata.get(
                    "file_name", ""
                ) or os.path.basename(file_path)
                processed_docs.append(doc)

            # Get the existing index
            index = self.indexes["manuals"]

            # Insert the documents using consistent method
            for doc in processed_docs:
                index.insert(doc)

            print("Documents added to the 'manuals' vector index.")
            return True
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            return False

    def _create_agent(self):
        tools = []
        tools.append(duckduckgo_search_tool())
        for index_name, index in self.indexes.items():
            query_engine = index.as_query_engine(similarity_top_k=10)
            tool = QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name=f"{index_name}_search",
                    description=f"Search the {index_name} for detailed automotive technical information and repair procedures.",
                ),
            )
            tools.append(tool)

        self.agent = ReActAgent.from_tools(
            tools,
            llm=Settings.llm,
            verbose=True,
            system_prompt=self.system_prompt,
            memory=ChatMemoryBuffer.from_defaults(token_limit=4096),
        )
        react_system_prompt = PromptTemplate(self.system_prompt)
        self.agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})

    def query(self, query: str, session_id: str) -> QueryResult:
        if not self.agent:
            raise ValueError(
                "Agent not created. There might be an issue with index loading or creation."
            )

        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append({"role": "user", "content": query})

        response = self.agent.chat(query)
        source_nodes = [
            {"text": str(node.node.text), "score": str(node.score)}
            for node in response.source_nodes
        ]

        self.sessions[session_id].append(
            {"role": "assistant", "content": response.response}
        )

        return QueryResult(answer=response.response, source_nodes=source_nodes)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self.sessions.get(session_id, [])

    def list_manuals(self) -> List[Dict[str, str]]:
        try:
            response = self.qdrant_client.scroll(
                collection_name="manuals", with_payload=True
            )
            unique_filepaths = {}
            for record in response[0]:
                if record.payload:
                    file_path = record.payload.get("file_path")
                    file_name = record.payload.get("file_name")
                    if file_path and file_path not in unique_filepaths:
                        unique_filepaths[file_path] = file_name

            manuals_list = [{"file_name": name} for _, name in unique_filepaths.items()]
            return manuals_list
        except Exception as e:
            print(f"Error listing manuals: {str(e)}")
            return []
