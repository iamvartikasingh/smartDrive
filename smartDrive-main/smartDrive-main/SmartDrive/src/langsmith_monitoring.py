"""
DriveSmart AI - Part 3: LangSmith Monitoring
Performance tracking and bottleneck identification
Save as: langsmith_monitoring_new.py
"""

import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv
import json

# Load environment with override
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path, override=True)

# LangSmith setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "DriveSmart-AI"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# Chroma import: try modular package, then consolidated langchain
try:
    from langchain_chroma import Chroma
except Exception:
    try:
        from langchain.vectorstores import Chroma
    except Exception:
        try:
            from langchain.vectorstores.chroma import Chroma
        except Exception:
            Chroma = None
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import chromadb

# ============================================================================
# PERFORMANCE MONITOR CLASS
# ============================================================================

class DriveSmartPerformanceMonitor:
    """Monitor and analyze DriveSmart AI performance"""
    
    def __init__(self):
        self.metrics = {
            'query_times': [],
            'answer_lengths': [],
            'source_counts': [],
            'errors': [],
            'query_log': []
        }
    
    def log_query(self, query: str, response_time: float, answer: str, 
                  sources: int, error: str = None):
        """Log a query execution"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response_time': response_time,
            'answer_length': len(answer) if answer else 0,
            'sources_count': sources,
            'error': error
        }
        
        self.metrics['query_times'].append(response_time)
        self.metrics['answer_lengths'].append(len(answer) if answer else 0)
        self.metrics['source_counts'].append(sources)
        self.metrics['query_log'].append(log_entry)
        
        if error:
            self.metrics['errors'].append(error)
    
    def calculate_statistics(self) -> Dict:
        """Calculate performance statistics"""
        
        if not self.metrics['query_times']:
            return {}
        
        query_times = self.metrics['query_times']
        
        stats = {
            'total_queries': len(query_times),
            'avg_response_time': sum(query_times) / len(query_times),
            'min_response_time': min(query_times),
            'max_response_time': max(query_times),
            'avg_answer_length': sum(self.metrics['answer_lengths']) / len(self.metrics['answer_lengths']),
            'avg_sources': sum(self.metrics['source_counts']) / len(self.metrics['source_counts']),
            'error_rate': len(self.metrics['errors']) / len(query_times) if query_times else 0
        }
        
        return stats
    
    def identify_bottlenecks(self) -> List[Dict]:
        """Identify performance bottlenecks"""
        
        bottlenecks = []
        stats = self.calculate_statistics()
        
        # Slow response time
        if stats.get('avg_response_time', 0) > 3.0:
            bottlenecks.append({
                'type': 'SLOW_RESPONSE',
                'severity': 'HIGH',
                'description': f"Average response time is {stats['avg_response_time']:.2f}s (target: <2s)",
                'recommendation': "Optimize retrieval, reduce k value, or implement caching"
            })
        
        # Insufficient sources
        if stats.get('avg_sources', 0) < 2:
            bottlenecks.append({
                'type': 'LOW_RETRIEVAL',
                'severity': 'MEDIUM',
                'description': f"Average sources retrieved: {stats['avg_sources']:.1f} (target: 3)",
                'recommendation': "Expand dataset or adjust search parameters"
            })
        
        # High error rate
        if stats.get('error_rate', 0) > 0.1:
            bottlenecks.append({
                'type': 'HIGH_ERRORS',
                'severity': 'CRITICAL',
                'description': f"Error rate: {stats['error_rate']*100:.1f}%",
                'recommendation': "Review error logs and add error handling"
            })
        
        # Inconsistent response length
        answer_lengths = self.metrics['answer_lengths']
        if answer_lengths:
            variance = max(answer_lengths) - min(answer_lengths)
            if variance > 500:
                bottlenecks.append({
                    'type': 'INCONSISTENT_OUTPUT',
                    'severity': 'LOW',
                    'description': f"Answer length varies by {variance} chars",
                    'recommendation': "Add response length constraints to prompts"
                })
        
        return bottlenecks
    
    def generate_report(self) -> str:
        """Generate comprehensive monitoring report"""
        
        stats = self.calculate_statistics()
        bottlenecks = self.identify_bottlenecks()
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║          DRIVESMART AI - PERFORMANCE MONITORING REPORT               ║
╚══════════════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

─────────────────────────────────────────────────────────────────────
📊 PERFORMANCE METRICS
─────────────────────────────────────────────────────────────────────
Total Queries:          {stats.get('total_queries', 0)}
Average Response Time:  {stats.get('avg_response_time', 0):.3f}s
Min Response Time:      {stats.get('min_response_time', 0):.3f}s
Max Response Time:      {stats.get('max_response_time', 0):.3f}s
Average Answer Length:  {stats.get('avg_answer_length', 0):.0f} characters
Average Sources Used:   {stats.get('avg_sources', 0):.1f}
Error Rate:             {stats.get('error_rate', 0)*100:.2f}%

─────────────────────────────────────────────────────────────────────
🔍 BOTTLENECKS IDENTIFIED: {len(bottlenecks)}
─────────────────────────────────────────────────────────────────────
"""
        
        if not bottlenecks:
            report += "✓ No significant bottlenecks detected. System performing well!\n"
        else:
            for i, bottleneck in enumerate(bottlenecks, 1):
                report += f"""
{i}. {bottleneck['type']} [{bottleneck['severity']}]
   • Description: {bottleneck['description']}
   • Recommendation: {bottleneck['recommendation']}
"""
        
        report += f"""
─────────────────────────────────────────────────────────────────────
💡 OPTIMIZATION RECOMMENDATIONS
─────────────────────────────────────────────────────────────────────
1. Implement caching for frequently asked questions
2. Add response time monitoring alerts (threshold: 2.5s)
3. Expand traffic law dataset for better coverage
4. Optimize prompt templates based on answer quality
5. Add user feedback collection mechanism

─────────────────────────────────────────────────────────────────────
📝 DETAILED QUERY LOG
─────────────────────────────────────────────────────────────────────
"""
        
        for i, log in enumerate(self.metrics['query_log'][:10], 1):  # Show last 10
            status = "✓" if not log['error'] else "✗"
            report += f"{i}. {status} [{log['response_time']:.2f}s] {log['query'][:50]}...\n"
        
        report += "\n" + "=" * 70 + "\n"
        
        return report
    
    def save_metrics(self, filename: str = "performance_metrics.json"):
        """Save metrics to file"""
        
        output_dir = Path(__file__).parent.parent / 'logs'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump({
                'statistics': self.calculate_statistics(),
                'bottlenecks': self.identify_bottlenecks(),
                'query_log': self.metrics['query_log']
            }, f, indent=2)
        
        print(f"✓ Metrics saved to: {output_path}")

# ============================================================================
# TEST WORKFLOW WITH MONITORING
# ============================================================================

def initialize_workflow():
    """Initialize workflow with monitoring"""
    
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=api_key
    )
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    )
    
    chroma_client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DB")
    )
    
    vectorstore = Chroma(
        client=chroma_client,
        collection_name="traffic_laws",
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Build LCEL chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a certified traffic law instructor.

Context: {context}

Question: {question}

Provide a clear, accurate answer.

Answer:"""
    )
    
    chain = RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough()
    }).assign(
        answer=lambda x: (
            prompt 
            | llm 
            | StrOutputParser()
        ).invoke({
            "context": format_docs(x["context"]),
            "question": x["question"]
        }),
        sources=lambda x: x["context"]
    )
    
    return chain

def run_monitored_evaluation():
    """Run evaluation with performance monitoring"""
    
    print("=" * 70)
    print("Running Monitored Evaluation")
    print("=" * 70)
    
    # Initialize
    monitor = DriveSmartPerformanceMonitor()
    chain = initialize_workflow()
    
    # Test queries
    test_queries = [
        "What are the penalties for speeding in Massachusetts?",
        "Can I use my phone at a red light?",
        "What happens if I get a DUI in California?",
        "Is it illegal to park in a handicapped spot?",
        "What are school zone speed limits?",
        "How much is a red light ticket?",
        "Can commercial trucks park on residential streets?",
        "What are the penalties for reckless driving?",
        "Do I need to wear a seatbelt in the back seat?",
        "What is the penalty for an expired license?"
    ]
    
    print(f"\nTesting {len(test_queries)} queries...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"[{i}/{len(test_queries)}] {query[:50]}...")
        
        start_time = time.time()
        error = None
        answer = ""
        sources = 0
        
        try:
            result = chain.invoke(query)
            answer = result['answer']
            sources = len(result['sources'])
        except Exception as e:
            error = str(e)
            print(f"  ✗ Error: {error}")
        
        response_time = time.time() - start_time
        
        # Log metrics
        monitor.log_query(query, response_time, answer, sources, error)
        
        print(f"  ✓ {response_time:.2f}s, {len(answer)} chars, {sources} sources")
    
    print("\n" + "=" * 70)
    print("Generating Performance Report...")
    print("=" * 70)
    
    # Generate and print report
    report = monitor.generate_report()
    print(report)
    
    # Save metrics
    monitor.save_metrics()
    
    return monitor

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    monitor = run_monitored_evaluation()
    
    print("\n✅ Monitoring complete!")
    print("Check logs/ folder for detailed metrics")