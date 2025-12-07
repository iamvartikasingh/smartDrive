"""
DriveSmart AI: Complete Implementation with Cloud ChromaDB
Connects to your cloud ChromaDB instance using credentials from src/.env
Data location: smartdrive/data/traffic_laws_dataset.json
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import json
import pandas as pd 
 # Add pandas for CSV reading
import re
from typing import List
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# ChromaDB Cloud Client
import chromadb

# Text processing
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store
from langchain_chroma import Chroma

# Embeddings - OpenAI text-embedding-3-small
from langchain_openai import OpenAIEmbeddings

# LLM - OpenAI GPT-4 (instead of Anthropic)
from langchain_openai import ChatOpenAI

# Core LangChain components
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda
)

# ============================================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================================
US_STATES = {
    "massachusetts": "Massachusetts",
    "ma": "Massachusetts",
    "california": "California",
    "ca": "California",
    "new york": "New York",
    "ny": "New York",
    "texas": "Texas",
    "tx": "Texas",
    "fl": "Florida"
}

def extract_jurisdictions(text: str) -> List[str]:
    if not text:
        return []

    t = text.lower()
    found = []

    # match multi-word and abbreviations safely
    for k, v in US_STATES.items():
        if re.search(r"\b" + re.escape(k) + r"\b", t):
            found.append(v)

    # dedupe preserve order
    seen = set()
    out = []
    for s in found:
        if s not in seen:
            seen.add(s)
            out.append(s)

    return out
# Load from src/.env file
env_path = Path(__file__).parent.parent / ".env"
print(env_path)
if env_path.exists():
    load_dotenv(env_path,override=True)
    # print(f"✓ Loaded environment from: {env_path}")
else:
    print(f"⚠ Warning: {env_path} not found, using system environment variables")
load_dotenv()

# Verify all required keys are present
REQUIRED_ENV_VARS = [
    "CHROMA_API_KEY",
    "CHROMA_TENANT", 
    "CHROMA_DB",
    "OPENAI_API_KEY"
]

missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}\nCheck your src/.env file")

print("✓ All required environment variables loaded")

# ============================================================================
# CLOUD CHROMADB VECTOR STORE
# ============================================================================

class CloudTrafficLawVectorStore:
    """Manages Cloud ChromaDB vector store for traffic laws"""
    
    def __init__(self):
        #load all keys from env :
        # env_path = Path(__file__).parent.parent/'src'/'.env'
        # print(env_path)
        # if env_path.exists():
        # load_dotenv()
        print(f"OPENAI_API_KEY from env: {os.getenv('OPENAI_API_KEY')}")
        print(f"Length: {len(os.getenv('OPENAI_API_KEY') or '')}")
        
        apikey=os.getenv("OPENAI_API_KEY")
        print("openai api key -  ",apikey)
        print("Type : ",type(apikey))
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=apikey        
        )
        print("self embeddings - ",self.embeddings)
        # Connect to ChromaDB Cloud
        self.chroma_client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DB")
        )
        
        print(f"✓ Connected to ChromaDB Cloud")
        print(f"  Tenant: {os.getenv('CHROMA_TENANT')}")
        print(f"  Database: {os.getenv('CHROMA_DB')}")
        
        self.vectorstore = None
    
    def load_data(self, csv_file):
        print("Data file path- ",csv_file)
        """Load traffic law data from JSON - uses smartdrive/data folder"""
        data_path = Path(csv_file)
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Traffic law dataset not found at: {data_path}\n"
                f"Please ensure the file exists at: smartdrive/data/traffic_laws_dataset.json"
            )
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        print(f"✓ Loaded data from: {data_path}")
        return data
    
    def prepare_documents(self, data: List[Dict]) -> List[Document]:
        """Convert traffic law data to LangChain documents"""
        documents = []
        
        for item in data:
            content = f"""
Jurisdiction: {item['jurisdiction']}
Category: {item['category']}
Violation: {item['violation']}
Law: {item['law_text']}
Statute: {item['statute']}
Penalty: {item['penalty']}
Severity: {item['severity']}
Prevention: {item['preventive_tip']}
Keywords: {', '.join(item['keywords'])}
"""
            
            metadata = {
                'id': item['id'],
                'jurisdiction': item['jurisdiction'],
                'category': item['category'],
                'violation': item['violation'],
                'statute': item['statute'],
                'severity': item['severity']
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    
    def initialize_vectorstore(self, documents: List[Document], collection_name="traffic_laws"):
        """Create and populate Cloud ChromaDB vectorstore"""
        
        print(f"\n[Creating Collection: {collection_name}]")
        
        # Create vectorstore with cloud client
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            client=self.chroma_client,
            collection_name=collection_name
        )
        
        print(f"✓ Cloud ChromaDB initialized with {len(documents)} documents")
        print(f"✓ Collection '{collection_name}' created successfully")
        
        return self.vectorstore
    
    def get_existing_vectorstore(self, collection_name="traffic_laws"):
        """Connect to existing collection in Cloud ChromaDB"""
        
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        
        print(f"✓ Connected to existing collection: {collection_name}")
        return self.vectorstore
    
    # def query(self, question: str, prompt_type: str = 'general'):
    #   chain = self.create_chain(prompt_type)  # always rebuild
# ============================================================================
# MODERN DRIVESMART WORKFLOW
# ============================================================================

class ModernDriveSmartWorkflow:
    """Complete workflow using modern langchain_core with Cloud ChromaDB"""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        
        # Use OpenAI GPT-4 as LLM
        self.llm = ChatOpenAI(
            model="gpt-4o",  # or "gpt-4-turbo" or "gpt-3.5-turbo" for cheaper
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
       
        self.chains = {}
        
        # Create the three prompts
        self.create_all_prompts()
    
    def create_all_prompts(self):
        """Create three different prompt templates"""
        
        self.prompts = {
            'general': PromptTemplate(
                input_variables=["context", "question"],
                template="""You are a certified traffic law instructor.

Use ONLY the context.

Answer the question in ONE short sentence.
Do not include numbering, labels, headings, or extra explanation.
If the answer is not in the context, reply exactly:
Not found in database.

Context:
{context}

Question:
{question}

Answer:"""
            ),      
    
    "scenario": PromptTemplate(
        input_variables=["context", "scenario"],
       template="""You are a certified traffic law instructor.

Use ONLY the context.

Answer the scenario in ONE short sentence.
Do not include numbering, labels, headings, or extra explanation.
If the answer is not in the context, reply exactly:
Not found in database.

Context:
{context}

Scenario:
{scenario}

Answer:"""

    ),
  
"comparative": PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a certified traffic law instructor.

Use ONLY the context.

Answer in 2 short sentences:
- sentence 1: Massachusetts rule/penalty
- sentence 2: California rule/penalty
If the answer is not in the context, reply exactly:
Not found in database.

Context:
{context}

Question:
{question}

Answer:"""

    )
        }
    
    def format_docs(self, docs):
        """Format documents for context"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def create_chain(self, prompt_type: str = 'general', retriever=None):
        """Create a retrieval chain for the specified prompt type"""
        aliases = {
        "general": "general",
        "default": "general",
        "scenario": "scenario",
        "scenerio": "scenario",   # common typo
        "case": "scenario",
        "comparative": "comparative",
        "compare": "comparative"
        }

        prompt_type = aliases.get(prompt_type, prompt_type)

        if prompt_type not in self.prompts:
           prompt_type = "general"
        if prompt_type not in self.prompts:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        prompt = self.prompts[prompt_type]
        input_var = "question" if "question" in prompt.input_variables else "scenario"

        # fallback retriever if not provided
        retriever = retriever or self.vectorstore.as_retriever(search_kwargs={"k": 3})

        chain = RunnableParallel(
            {
                "context": retriever,
                input_var: RunnablePassthrough()
            }
        ).assign(
            answer=lambda x: (
                prompt
                | self.llm
                | StrOutputParser()
            ).invoke({
                "context": self.format_docs(x["context"]),
                input_var: x[input_var]
            }),
            sources=lambda x: x["context"]
        )

        return chain
    def query(self, question: str, prompt_type: str = "general"):
        states = extract_jurisdictions(question)

        if states:
            docs = []
            for st in states:
                r = self.vectorstore.as_retriever(
                    search_kwargs={"k": 4, "filter": {"jurisdiction": st}}
                )
                docs.extend(r.invoke(question))
            # de-dupe by id if needed
            seen = set()
            filtered = []
            for d in docs:
                _id = d.metadata.get("id")
                if _id and _id in seen:
                    continue
                seen.add(_id)
                filtered.append(d)
            retriever = RunnableLambda(lambda _: filtered)
        else:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 8})

        chain = self.create_chain(prompt_type, retriever=retriever)
        result = chain.invoke(question)

        raw = result["answer"].strip()
        first_line = next((l.strip() for l in raw.splitlines() if l.strip()), "")
        first_line = re.sub(r"^\s*\d+[\.\)]\s*", "", first_line)
        first_line = re.sub(r"^\s*direct answer\s*:\s*", "", first_line, flags=re.I)

        answer = first_line if first_line else "Not found in database"

        return {
            "answer": answer,
            "sources": result.get("sources", []),
            "detected_jurisdiction": ", ".join(states) if states else "Unspecified"
        }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution with Cloud ChromaDB - loads from smartdrive/data folder"""
    
    print("=" * 70)
    print("DriveSmart AI: Cloud ChromaDB Implementation")
    print("=" * 70)
    
    # Step 1: Connect to Cloud ChromaDB
    print("\n[1] Connecting to Cloud ChromaDB...")
    vector_store_manager = CloudTrafficLawVectorStore()
    
    # Step 2: Load data from smartdrive/data folder
    print("\n[2] Loading Traffic Law Data from smartdrive/data/...")
    data_path = Path(__file__).parent.parent/'data'/'traffic_laws_dataset.json'
    'smartdrive/data/traffic_laws_dataset.json'
    
    try:
        data = vector_store_manager.load_data(data_path)
        documents = vector_store_manager.prepare_documents(data)
        print(f"✓ Loaded {len(documents)} traffic law documents")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nExpected file structure:")
        print("  smartdrive/")
        print("    └── data/")
        print("        └── traffic_laws_dataset.json")
        print("\nPlease ensure the file exists at this location.")
        return None, None
    
    # Step 3: Initialize vectorstore (creates collection and uploads data)
    print("\n[3] Initializing Cloud Vector Store...")
    vectorstore = vector_store_manager.initialize_vectorstore(
        documents, 
        collection_name="traffic_laws"
    )
    
    # Step 4: Create workflow
    print("\n[4] Creating Modern Workflow...")
    workflow = ModernDriveSmartWorkflow(vectorstore)
    print("✓ Workflow initialized")
    
    # Step 5: Test queries
    print("\n[5] Testing Queries...")
    print("=" * 70)
    
    test_queries = [
        {
            'question': "Is using a phone at a red light a violation in Massachusetts?",
            'type': 'general'
        },
        {
            'question': "I was driving 45 mph in a residential area. What are the consequences?",
            'type': 'scenario'
        },
        {
            'question': "How do DUI penalties compare between Massachusetts and California?",
            'type': 'comparative'
        }
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i} ({query['type']}) ---")
        print(f"Question: {query['question']}")
        
        result = workflow.query(query['question'], query['type'])
        
        print(f"\nAnswer: {result['answer'][:300]}...")
        print(f"Sources: {len(result['sources'])} documents retrieved")
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("✅ Cloud ChromaDB Implementation Complete!")
    print("=" * 70)
    print("\nYour data is now stored in Cloud ChromaDB:")
    print(f"  - Tenant: {os.getenv('CHROMA_TENANT')}")
    print(f"  - Database: {os.getenv('CHROMA_DB')}")
    print(f"  - Collection: traffic_laws")
    print(f"  - Documents: {len(documents)}")
    
    return workflow, vectorstore

# ============================================================================
# HELPER: CONNECT TO EXISTING COLLECTION
# ============================================================================

def connect_to_existing_collection(collection_name="traffic_laws"):
    """Connect to an existing collection (if you've already uploaded data)"""
    
    print("=" * 70)
    print("Connecting to Existing Cloud ChromaDB Collection")
    print("=" * 70)
    
    vector_store_manager = CloudTrafficLawVectorStore()
    vectorstore = vector_store_manager.get_existing_vectorstore(collection_name)
    
    workflow = ModernDriveSmartWorkflow(vectorstore)
    
    print(f"\n✓ Connected to collection: {collection_name}")
    print("✓ Ready to query!")
    
    return workflow, vectorstore

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING FROM: smartdrive/data/cloud_chromadb_complete.py")
    print("=" * 70)
    
    # Verify file structure
    print("\n[Checking File Structure]")
    data_file = Path(__file__).parent.parent/'data'/'traffic_laws_dataset.json'
    
    if data_file.exists():
        print(f"✓ Found: {data_file}")
    else:
        print(f"❌ Missing: {data_file}")
        print("\nPlease create the file at: smartdrive/data/traffic_laws_dataset.json")
        exit(1)
    
    # First time: Initialize and upload data
    workflow, vectorstore = main()
    
    if workflow is None:
        print("\n❌ Setup failed. Please fix the errors above.")
        exit(1)
    
    # Subsequent times: Just connect to existing collection
    # Uncomment this line after first run:
    # workflow, vectorstore = connect_to_existing_collection("traffic_laws")
    
    # Test a query
    print("\n" + "=" * 70)
    print("Quick Test Query")
    print("=" * 70)
    
    test_question = "What are the penalties for speeding in Massachusetts?"
    result = workflow.query(test_question)
    
    print(f"\nQuestion: {test_question}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources used: {len(result['sources'])}")
    
    print("\n" + "=" * 70)
    print("✅ SUCCESS! Data is now in Cloud ChromaDB!")
    print("=" * 70)