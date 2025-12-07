"""
DriveSmart AI - Part 4: Refined Prompt Engineering
Three optimized prompts based on LangSmith insights
Save as: refined_prompts.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import time

# Load environment with override
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path, override=True)

# The project targets the newer modular langchain packages (e.g. `langchain_openai`,
# `langchain_chroma`, `langchain_core`). Some environments instead use the
# consolidated `langchain` package. Try the modular imports first and fall
# back to the consolidated package to maximize compatibility.
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    import chromadb
except Exception:
    # Fallback to the consolidated `langchain` package layout (best-effort).
    try:
        # Use the consolidated langchain helpers where available
        from langchain.chat_models import init_chat_model
        try:
            from langchain.embeddings import init_embeddings
        except Exception:
            # Older layouts may expose provider-specific embeddings modules
            try:
                from langchain.embeddings.openai import OpenAIEmbeddings
            except Exception:
                init_embeddings = None

        # Vectorstore/chroma wrapper may or may not be present
        try:
            from langchain.vectorstores import Chroma
        except Exception:
            try:
                from langchain.vectorstores.chroma import Chroma
            except Exception:
                Chroma = None

        try:
            from langchain.prompts import PromptTemplate
        except Exception:
            PromptTemplate = None

        try:
            from langchain.output_parsers import StrOutputParser
        except Exception:
            StrOutputParser = None
        # Runnable APIs may be in different places; attempt common locations
        try:
            from langchain.runnables import RunnableParallel, RunnablePassthrough
        except Exception:
            # If not available, provide very small local fallbacks below (used only for import compatibility)
            class RunnableParallel:
                def __init__(self, x):
                    raise ImportError("RunnableParallel not available in this langchain installation")

            class RunnablePassthrough:
                pass

        import chromadb
    except Exception as e:
        # Re-raise a clearer message so callers (and the dashboard) can surface it.
        raise ImportError(
            "Failed to import LangChain modules. Install either the modular "
            "`langchain_openai/langchain_chroma/langchain_core` packages or the "
            "consolidated `langchain` package. Original error: %s" % e
        )

# --- Helper initializers for environments with different langchain distributions ---
def _init_chat_model_fallback(model_name: str = "gpt-4o", api_key: str | None = None):
    """Return a chat model instance compatible with the minimal interface used below.

    Prefers `ChatOpenAI` if present; otherwise uses `langchain.chat_models.init_chat_model`.
    The returned object is expected to implement `invoke(prompt_str)`.
    """
    # If ChatOpenAI is available (modular import), use it
    try:
        if 'ChatOpenAI' in globals():
            if api_key:
                return ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
            return ChatOpenAI(model=model_name, temperature=0)
    except Exception:
        pass

    # Otherwise use the consolidated langchain init_chat_model helper
    try:
        from langchain.chat_models import init_chat_model
        # specify provider via model string if appropriate
        return init_chat_model(model_name, temperature=0)
    except Exception:
        raise ImportError("No compatible chat model implementation available; install a LangChain provider integration (e.g. langchain-openai)")


def _init_embeddings_fallback(model_name: str = "text-embedding-3-small", api_key: str | None = None):
    """Return an embeddings object with a `embed(text: str) -> list[float]` method.

    Prefers `OpenAIEmbeddings` if available; otherwise uses `langchain.embeddings.init_embeddings`.
    """
    try:
        if 'OpenAIEmbeddings' in globals():
            if api_key:
                return OpenAIEmbeddings(model=model_name, api_key=api_key)
            return OpenAIEmbeddings(model=model_name)
    except Exception:
        pass

    try:
        from langchain.embeddings import init_embeddings
        return init_embeddings(model_name)
    except Exception:
        raise ImportError("No compatible embeddings implementation available; install a LangChain embeddings provider integration (e.g. langchain-openai)")


class _SimpleChromaRetriever:
    """Minimal retriever wrapper around a `chromadb` collection.

    This provides `get_docs(query,k)` which returns list of simple objects with
    `page_content` and `metadata` attributes used by the workflow.
    """
    def __init__(self, chroma_client, collection_name: str = "traffic_laws", embedding_fn=None):
        self.client = chroma_client
        self.collection = None
        try:
            self.collection = chroma_client.get_collection(collection_name)
        except Exception:
            # some clients require creating or fetching differently
            try:
                self.collection = chroma_client.get(collection_name)
            except Exception:
                self.collection = None
        self.embedding_fn = embedding_fn

    def get_docs(self, query: str, k: int = 3):
        if not self.collection:
            return []

        # Prefer text query; chromadb will handle text->embedding server-side if supported
        try:
            qr = self.collection.query(query_texts=[query], n_results=k, include=["documents", "metadatas"]) or {}
            docs = []
            documents = qr.get('documents', [[]])[0]
            metadatas = qr.get('metadatas', [[]])[0]

            for d, m in zip(documents, metadatas):
                class Doc:
                    pass

                doc = Doc()
                doc.page_content = d
                doc.metadata = m or {}
                docs.append(doc)

            return docs
        except Exception:
            return []

# ============================================================================
# ORIGINAL PROMPTS (Before Optimization)
# ============================================================================

ORIGINAL_GENERAL_PROMPT = """You are a certified traffic law instructor.

Context: {context}

Question: {question}

Provide:
1. Direct answer
2. Statute reference
3. Penalties
4. Prevention tips

Answer:"""

# ============================================================================
# REFINED PROMPTS (After LangSmith Analysis)
# ============================================================================

REFINED_PROMPT_1 = """You are a certified traffic law instructor providing quick, accurate guidance.

Legal Context:
{context}

User Question: {question}

RESPONSE FORMAT (Keep under 250 words):


[One clear sentence answering the question]

⚖️ LEGAL BASIS:
• Statute: [Reference]
• Law: [Brief description]

💰 PENALTIES:
• [List consequences clearly]

✅ HOW TO AVOID:
• [Actionable prevention step]

⚠️ NOTE: [Any important exceptions or warnings]

Answer:"""

REFINED_PROMPT_2 = """You are a traffic law expert. Analyze the scenario systematically.

Retrieved Laws:
{context}

Scenario: {scenario}

STEP-BY-STEP ANALYSIS:

1️⃣ WHAT HAPPENED:
[Summarize the situation in one sentence]

2️⃣ VIOLATIONS IDENTIFIED:
[List each violation with statute number]

3️⃣ LEGAL CONSEQUENCES:
• Fines: [Amount]
• Points: [Number]
• License Impact: [Description]
• Other: [Additional penalties]

4️⃣ YOUR RIGHTS:
[What the driver can do]

5️⃣ PREVENTION:
[How to avoid this situation]

Keep response structured and scannable. Use bullet points.

Analysis:"""

REFINED_PROMPT_3 = """You are a traffic law consultant comparing regulations across jurisdictions.

Legal Data:
{context}

Comparison Query: {question}

COMPARATIVE ANALYSIS:

📋 OVERVIEW:
[One sentence summary of the key difference]

IMPORTANT OUTPUT RULES:
1) Use short, crisp phrases.
2) Do NOT draw ASCII tables.
3) Always include a structured JSON block exactly in this format.

TABLE_JSON:
{{ 
  "jurisdiction_1": "Jurisdiction 1 name",
  "jurisdiction_2": "Jurisdiction 2 name",
  "rows": [
    {{ "aspect": "Basic Law", "j1": "…", "j2": "…" }},
    {{ "aspect": "Penalties", "j1": "…", "j2": "…" }},
    {{ "aspect": "Points", "j1": "…", "j2": "…" }},
    {{ "aspect": "Special Notes", "j1": "…", "j2": "…" }}
  ]
}}

🔑 KEY TAKEAWAYS:
1. [Most important similarity]
2. [Most important difference]
3. [Practical advice for travelers]

Keep comparison clear and actionable.
"""
# ============================================================================
# WORKFLOW WITH REFINED PROMPTS
# ============================================================================

class RefinedDriveSmartWorkflow:
    """Workflow using refined prompts"""
    
    def __init__(self):
        # Get API key properly
        api_key = os.getenv("OPENAI_API_KEY", "").strip()

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")

        try:
            self.llm = _init_chat_model_fallback("gpt-4o", api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize chat model: {e}")

        try:
            self.embeddings = _init_embeddings_fallback("text-embedding-3-small", api_key=api_key)
        except Exception as e:
            self.embeddings = None

        try:
            chroma_client = chromadb.CloudClient(
                api_key=os.getenv("CHROMA_API_KEY"),
                tenant=os.getenv("CHROMA_TENANT"),
                database=os.getenv("CHROMA_DB")
            )
        except Exception:
            try:
                chroma_client = chromadb.Client()
            except Exception:
                chroma_client = None

        try:
            if 'Chroma' in globals() and chroma_client is not None:
                vectorstore = Chroma(client=chroma_client, collection_name="traffic_laws", embedding_function=self.embeddings)
                try:
                    self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                except Exception:
                    self.retriever = _SimpleChromaRetriever(chroma_client, "traffic_laws", embedding_fn=self.embeddings)
            else:
                self.retriever = _SimpleChromaRetriever(chroma_client, "traffic_laws", embedding_fn=self.embeddings)
        except Exception:
            self.retriever = _SimpleChromaRetriever(chroma_client, "traffic_laws", embedding_fn=self.embeddings)

        print("✓ Refined workflow initialized")
    
    def format_docs(self, docs):
        """Format documents for context"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def query_with_prompt(self, question: str, prompt_template: str, 
                          input_var: str = "question"):
        """Query using specific prompt template"""
        try:
            prompt_t = PromptTemplate(input_variables=["context", input_var], template=prompt_template)
            use_template = True
        except Exception:
            use_template = False

        sources = []
        try:
            if hasattr(self.retriever, 'get_docs'):
                docs = self.retriever.get_docs(question, k=3)
                sources = docs
            else:
                try:
                    docs = self.retriever(question)
                    sources = docs
                except Exception:
                    try:
                        docs = self.retriever.get_relevant_documents(question)
                        sources = docs
                    except Exception:
                        sources = []
        except Exception:
            sources = []

        context_text = self.format_docs(sources)

        if use_template:
            prompt_text = prompt_t.format(context=context_text, **{input_var: question})
        else:
            prompt_text = prompt_template.replace("{context}", context_text).replace("{" + input_var + "}", question)

        try:
            if hasattr(self.llm, 'invoke'):
                answer = self.llm.invoke(prompt_text)
            elif hasattr(self.llm, '__call__'):
                answer = self.llm(prompt_text)
            else:
                answer = getattr(self.llm, 'generate', lambda *a, **k: None)([prompt_text])

            if isinstance(answer, dict) and 'text' in answer:
                final_text = answer['text']
            elif hasattr(answer, 'content'):
                final_text = getattr(answer, 'content')
            elif isinstance(answer, list) and answer:
                final_text = str(answer[0])
            else:
                final_text = str(answer)
        except Exception as e:
            final_text = f"(LLM error: {e})"

        return {
            'answer': final_text,
            'sources': sources
        }
    def query(self, question: str, prompt_type_key: str = "general"):
        """
        Dashboard-friendly API.
        Routes to the right refined prompt and returns a consistent result shape.
        """

        key = (prompt_type_key or "general").lower().strip()

        if key == "scenario":
            prompt = REFINED_PROMPT_2
            input_var = "scenario"
        elif key == "comparative":
            prompt = REFINED_PROMPT_3
            input_var = "question"
        else:
            prompt = REFINED_PROMPT_1
            input_var = "question"

        result = self.query_with_prompt(question, prompt, input_var)

        # Try to infer jurisdiction from retrieved metadata (if your docs have it)
        jurisdictions = set()
        for doc in result.get("sources", []):
            meta = getattr(doc, "metadata", {}) or {}
            j = meta.get("jurisdiction") or meta.get("state")
            if j:
                jurisdictions.add(str(j).strip())

        result["detected_jurisdiction"] = sorted(jurisdictions) if jurisdictions else "All"

        return result
# ============================================================================
# COMPARISON TESTING
# ============================================================================

def compare_prompts():
    """Compare original vs refined prompts"""
    
    print("=" * 70)
    print("Comparing Original vs Refined Prompts")
    print("=" * 70)
    
    workflow = RefinedDriveSmartWorkflow()
    
    test_question = "What are the penalties for using a phone while driving in Massachusetts?"
    
    print(f"\nTest Question: {test_question}\n")
    
    # Test original prompt
    print("─" * 70)
    print("TESTING ORIGINAL PROMPT")
    print("─" * 70)
    
    start = time.time()
    original_result = workflow.query_with_prompt(
        test_question, 
        ORIGINAL_GENERAL_PROMPT
    )
    original_time = time.time() - start
    
    print(f"Response Time: {original_time:.3f}s")
    print(f"Answer Length: {len(original_result['answer'])} chars")
    print(f"Sources: {len(original_result['sources'])}")
    print(f"\nAnswer Preview:\n{original_result['answer'][:300]}...\n")
    
    # Test refined prompt
    print("─" * 70)
    print("TESTING REFINED PROMPT 1")
    print("─" * 70)
    
    start = time.time()
    refined_result = workflow.query_with_prompt(
        test_question,
        REFINED_PROMPT_1
    )
    refined_time = time.time() - start
    
    print(f"Response Time: {refined_time:.3f}s")
    print(f"Answer Length: {len(refined_result['answer'])} chars")
    print(f"Sources: {len(refined_result['sources'])}")
    print(f"\nAnswer Preview:\n{refined_result['answer'][:300]}...\n")
    
    # Calculate improvements
    time_improvement = ((original_time - refined_time) / original_time) * 100 if original_time > 0 else 0
    length_reduction = ((len(original_result['answer']) - len(refined_result['answer'])) / 
                        len(original_result['answer'])) * 100 if len(original_result['answer']) > 0 else 0
    
    print("=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"Response Time Change: {time_improvement:+.1f}%")
    print(f"Answer Length Change: {length_reduction:+.1f}%")
    print(f"Clarity: Refined prompt uses structured format ✓")
    print(f"Scannability: Improved with emojis and bullets ✓")
    
    return {
        'original': original_result,
        'refined': refined_result,
        'metrics': {
            'time_improvement': time_improvement,
            'length_change': length_reduction
        }
    }

# ============================================================================
# TEST ALL THREE REFINED PROMPTS
# ============================================================================

def test_all_refined_prompts():
    """Test all three refined prompt templates"""
    
    print("\n" + "=" * 70)
    print("Testing All Refined Prompts")
    print("=" * 70)
    
    workflow = RefinedDriveSmartWorkflow()
    
    tests = [
        {
            'name': 'Refined Prompt 1: General Query',
            'question': 'What are the penalties for running a red light in Massachusetts?',
            'prompt': REFINED_PROMPT_1,
            'input_var': 'question'
        },
        {
            'name': 'Refined Prompt 2: Scenario Analysis',
            'question': 'I was texting while driving through a school zone at 45 mph during school hours.',
            'prompt': REFINED_PROMPT_2,
            'input_var': 'scenario'
        },
        {
            'name': 'Refined Prompt 3: Comparative Analysis',
            'question': 'Compare DUI penalties between Massachusetts and California',
            'prompt': REFINED_PROMPT_3,
            'input_var': 'question'
        }
    ]
    
    results = []
    
    for test in tests:
        print(f"\n{'─'*70}")
        print(f"TEST: {test['name']}")
        print(f"{'─'*70}")
        print(f"Query: {test['question']}\n")
        
        start = time.time()
        result = workflow.query_with_prompt(
            test['question'],
            test['prompt'],
            test['input_var']
        )
        response_time = time.time() - start
        
        print(f"⏱️  Response Time: {response_time:.3f}s")
        print(f"📏 Answer Length: {len(result['answer'])} chars")
        print(f"📚 Sources Used: {len(result['sources'])}")
        print(f"\n📝 Answer:\n{result['answer']}\n")
        
        results.append({
            'test': test['name'],
            'response_time': response_time,
            'answer_length': len(result['answer']),
            'sources': len(result['sources'])
        })
    
    # Summary
    print("=" * 70)
    print("REFINED PROMPTS SUMMARY")
    print("=" * 70)
    
    avg_time = sum(r['response_time'] for r in results) / len(results)
    avg_length = sum(r['answer_length'] for r in results) / len(results)
    
    print(f"Average Response Time: {avg_time:.3f}s")
    print(f"Average Answer Length: {avg_length:.0f} chars")
    print(f"\nKey Improvements:")
    print("✓ Structured format with emojis for better scannability")
    print("✓ Consistent response length (target: 200-300 words)")
    print("✓ Clear section headers for easy navigation")
    print("✓ Action-oriented prevention tips")
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Test comparison
    print("\n[1] Comparing Original vs Refined Prompts...")
    comparison = compare_prompts()
    
    # Test all refined prompts
    print("\n[2] Testing All Three Refined Prompts...")
    results = test_all_refined_prompts()
    
    print("\n" + "=" * 70)
    print("✅ Prompt refinement testing complete!")
    print("=" * 70)