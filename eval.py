import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import json
import os
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
import time
import argparse


class SearchBenchmark:
    def __init__(self, db_path='docs.db', model_name='all-MiniLM-L6-v2'):
        self.db_path = db_path
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"Embedding model loaded successfully")
        self._query_embedding_cache = {}
        self._hybrid_cache = {}
        self.test_queries = []
        self.results = {}
        
        # Load or create test dataset
        self.test_dataset_path = 'search_test_dataset.json'
        if os.path.exists(self.test_dataset_path):
            self.load_test_dataset()
        else:
            print("No test dataset found. Use create_test_dataset() to create one.")
    
    def load_database_content(self):
        """Load all documents and their embeddings from the database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Get all documents
        c.execute("SELECT id, title, content, link FROM documents")
        self.documents = {}
        for doc_id, title, content, link in c.fetchall():
            doc_text = title
            if content and content != "Linked document":
                doc_text += " " + content
            self.documents[str(doc_id)] = {
                "id": str(doc_id),
                "title": title,
                "content": content,
                "link": link,
                "text": doc_text,
                "type": "document"
            }
        
        # Get all tasks
        c.execute("SELECT id, title, description FROM tasks")
        for task_id, title, description in c.fetchall():
            doc_id = f"task_{task_id}"
            self.documents[doc_id] = {
                "id": doc_id,
                "title": title,
                "description": description,
                "text": f"{title} {description}",
                "type": "task"
            }
            
        # Get all embeddings
        c.execute("SELECT doc_id, embedding FROM embeddings")
        self.embeddings = {}
        for doc_id, emb_bytes in c.fetchall():
            self.embeddings[doc_id] = np.frombuffer(emb_bytes, dtype=np.float32)
        
        conn.close()
        
        # Create BM25 index
        self.tokenized_corpus = []
        self.doc_ids = []
        
        for doc_id, doc in self.documents.items():
            self.tokenized_corpus.append(doc["text"].lower().split())
            self.doc_ids.append(doc_id)
            
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print(f"Loaded {len(self.documents)} documents and {len(self.embeddings)} embeddings")
    
    def create_test_dataset(self, num_queries=10):
        """
        Interactive tool to create a test dataset of queries and relevant documents
        """
        self.load_database_content()
        test_dataset = []
        
        print("\n=== TEST DATASET CREATION ===")
        print("This tool will help you create a test dataset for evaluating search quality.")
        print("For each query, you'll see the top results from both methods and can mark which ones are relevant.")
        
        for i in range(num_queries):
            query = input(f"\nEnter test query #{i+1}: ")
            if not query:
                break
                
            # Get results from both methods
            embedding_results = self.embedding_search(query, top_k=10)
            hybrid_results = self.hybrid_search(query, top_k=10)
            
            # Combine results for review
            combined_results = {}
            for doc_id, score in embedding_results:
                combined_results[doc_id] = {"doc_id": doc_id, "embedding_score": score, "hybrid_score": 0}
            
            for doc_id, score in hybrid_results:
                if doc_id in combined_results:
                    combined_results[doc_id]["hybrid_score"] = score
                else:
                    combined_results[doc_id] = {"doc_id": doc_id, "embedding_score": 0, "hybrid_score": score}
            
            # Display results for relevance judging
            print("\nPlease review these potential results and mark which ones are relevant (1=relevant, 0=not relevant):")
            results = []
            relevant_docs = []
            
            for j, (doc_id, scores) in enumerate(combined_results.items()):
                doc = self.documents.get(doc_id, {"title": "Unknown", "type": "unknown"})
                print(f"\n{j+1}. {doc.get('title', 'Unknown')} ({doc.get('type', 'unknown')})")
                print(f"   Embedding score: {scores['embedding_score']:.4f}, Hybrid score: {scores['hybrid_score']:.4f}")
                
                if doc.get('type') == 'document' and doc.get('content'):
                    preview = doc.get('content', '')[:100] + '...' if len(doc.get('content', '')) > 100 else doc.get('content', '')
                    print(f"   Preview: {preview}")
                elif doc.get('type') == 'task':
                    preview = doc.get('description', '')[:100] + '...' if len(doc.get('description', '')) > 100 else doc.get('description', '')
                    print(f"   Preview: {preview}")
                
                relevance = input(f"   Relevant (1/0)? ")
                if relevance == '1':
                    relevant_docs.append(doc_id)
                
                results.append({
                    "doc_id": doc_id,
                    "title": doc.get('title', 'Unknown'),
                    "type": doc.get('type', 'unknown'),
                    "embedding_score": scores["embedding_score"],
                    "hybrid_score": scores["hybrid_score"]
                })
            
            test_dataset.append({
                "query": query,
                "relevant_docs": relevant_docs,
                "results": results
            })
            
            print(f"\nQuery {i+1} completed. {len(relevant_docs)} documents marked as relevant.")
        
        # Save test dataset
        with open(self.test_dataset_path, 'w') as f:
            json.dump(test_dataset, f, indent=2)
        
        self.test_queries = test_dataset
        print(f"\nTest dataset with {len(test_dataset)} queries created and saved.")
    
    def load_test_dataset(self):
        """Load the test dataset from file"""
        with open(self.test_dataset_path, 'r') as f:
            self.test_queries = json.load(f)
        print(f"Loaded test dataset with {len(self.test_queries)} queries")
    
    def embedding_search(self, query, top_k=5, return_all=False):
        if query in self._query_embedding_cache:
            query_embedding = self._query_embedding_cache[query]
        else:
            query_embedding = self.model.encode(query)
            self._query_embedding_cache[query] = query_embedding

        # Precompute document embeddings matrix and norms
        if not hasattr(self, 'doc_embeddings_matrix') or not hasattr(self, 'doc_norms'):
            doc_ids = list(self.embeddings.keys())
            self.doc_embeddings_matrix = np.array([self.embeddings[did] for did in doc_ids])
            self.doc_norms = np.linalg.norm(self.doc_embeddings_matrix, axis=1)
            self.doc_ids = doc_ids  # Store order of doc_ids

        # Compute similarities in one vectorized operation
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            similarities = np.zeros(len(self.doc_ids))
        else:
            similarities = np.dot(self.doc_embeddings_matrix, query_embedding) / (self.doc_norms * query_norm)

        # Combine with doc_ids and sort
        results = list(zip(self.doc_ids, similarities))
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        return sorted_results if return_all else sorted_results[:top_k]
    
    def bm25_search(self, query, top_k=5, return_all=False):
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        results = [(doc_id, score) for doc_id, score in zip(self.doc_ids, bm25_scores)]
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        if return_all:
            return sorted_results
        return sorted_results[:top_k]
    
    def hybrid_search(self, query, top_k=5, semantic_weight=0.7, bm25_weight=0.3, method='score', k_rrf=60):
        # Cache for hybrid search results to avoid redundant computations
        if not hasattr(self, '_hybrid_cache'):
            self._hybrid_cache = {}
            
        # Create a cache key based on the parameters
        cache_key = f"{query}_{top_k}_{semantic_weight}_{method}_{k_rrf}"
        
        # Return cached results if available
        if cache_key in self._hybrid_cache:
            return self._hybrid_cache[cache_key][:top_k]
            
        if method == 'score':
            # Get full results for normalization
            embedding_results = self.embedding_search(query, return_all=True)
            bm25_results = self.bm25_search(query, return_all=True)

            # Convert to dictionaries and clip embedding scores to non-negative
            emb_scores = {doc_id: max(0, score) for doc_id, score in embedding_results}
            bm25_scores = {doc_id: score for doc_id, score in bm25_results}

            # Min-max normalization for embedding scores
            min_emb = min(emb_scores.values()) if emb_scores else 0
            max_emb = max(emb_scores.values()) if emb_scores else 1
            if max_emb > min_emb:
                normalized_emb = {doc_id: (score - min_emb) / (max_emb - min_emb) 
                                for doc_id, score in emb_scores.items()}
            else:
                normalized_emb = {doc_id: 0 for doc_id in emb_scores}

            # Min-max normalization for BM25 scores
            min_bm25 = min(bm25_scores.values()) if bm25_scores else 0
            max_bm25 = max(bm25_scores.values()) if bm25_scores else 1
            if max_bm25 > min_bm25:
                normalized_bm25 = {doc_id: (score - min_bm25) / (max_bm25 - min_bm25) 
                                for doc_id, score in bm25_scores.items()}
            else:
                normalized_bm25 = {doc_id: 0 for doc_id in bm25_scores}

            # Combine scores
            combined_scores = {}
            all_doc_ids = set(normalized_emb.keys()) | set(normalized_bm25.keys())
            for doc_id in all_doc_ids:
                score = (semantic_weight * normalized_emb.get(doc_id, 0) + 
                        bm25_weight * normalized_bm25.get(doc_id, 0))
                combined_scores[doc_id] = score

            # Store in cache and return top-k
            sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            self._hybrid_cache[cache_key] = sorted_results
            return sorted_results[:top_k]

        elif method == 'rrf':
            # Get full ranked lists
            embedding_results = self.embedding_search(query, return_all=True)
            bm25_results = self.bm25_search(query, return_all=True)

            # Create rank dictionaries (rank starts at 1)
            rank_emb = {doc_id: rank for rank, (doc_id, _) in enumerate(embedding_results, start=1)}
            rank_bm25 = {doc_id: rank for rank, (doc_id, _) in enumerate(bm25_results, start=1)}

            # Compute RRF scores
            rrf_scores = {}
            all_doc_ids = set(rank_emb.keys()) | set(rank_bm25.keys())
            for doc_id in all_doc_ids:
                score = 0
                if doc_id in rank_emb:
                    score += 1 / (k_rrf + rank_emb[doc_id])
                if doc_id in rank_bm25:
                    score += 1 / (k_rrf + rank_bm25[doc_id])
                rrf_scores[doc_id] = score

            # Store in cache and return top-k
            sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            self._hybrid_cache[cache_key] = sorted_results
            return sorted_results[:top_k]

        else:
            raise ValueError("Invalid method. Use 'score' or 'rrf'.")
    
    def evaluate_search(self, search_func, query, relevant_docs, top_k=5, name=""):
        start_time = time.time()
        results = search_func(query, top_k)  # Use positional argument
        end_time = time.time()
        
        # Calculate metrics (example implementation)
        retrieved_docs = [doc_id for doc_id, _ in results]
        precision = sum(1 for doc_id in retrieved_docs if doc_id in relevant_docs) / len(retrieved_docs) if retrieved_docs else 0
        recall = sum(1 for doc_id in retrieved_docs if doc_id in relevant_docs) / len(relevant_docs) if relevant_docs else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Mean Reciprocal Rank (MRR)
        mrr = 0
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                mrr = 1 / (i + 1)
                break
        
        # NDCG
        dcg = 0
        idcg = 0
        for i, doc_id in enumerate(retrieved_docs):
            relevance = 1 if doc_id in relevant_docs else 0
            dcg += relevance / np.log2(i + 2)  # +2 because log_2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        ideal_ranking = [1] * min(len(relevant_docs), top_k) + [0] * max(0, top_k - len(relevant_docs))
        for i, relevance in enumerate(ideal_ranking):
            idcg += relevance / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mrr": mrr,
            "ndcg": ndcg,
            "query_time": end_time - start_time
        }
    
    def run_benchmark(self, semantic_weights):
        self.load_database_content()
        from tqdm import tqdm
        import numpy as np

        # Initialize caches for improved performance
        self._query_embedding_cache = {}
        self._hybrid_cache = {}
        

        # Create matrix of embeddings
        self.doc_embeddings_matrix = np.array([emb for emb in self.embeddings.values()])
        self.doc_ids = list(self.embeddings.keys())
        self.doc_norms = np.linalg.norm(self.doc_embeddings_matrix, axis=1)

        # Test pure embedding search
        print("\nEvaluating pure embedding search...")
        embedding_results = []
        for query_data in tqdm(self.test_queries):
            query = query_data["query"]
            relevant_docs = query_data["relevant_docs"]
            metrics = self.evaluate_search(
                self.embedding_search,
                query,
                relevant_docs,
                name="Pure Embedding"
            )
            embedding_results.append(metrics)
        
        # Compute average metrics for embedding search
        avg_metrics = {metric: np.mean([r[metric] for r in embedding_results]) 
                    for metric in embedding_results[0].keys()}
        self.results["pure_embedding"] = {
            "per_query": embedding_results,
            "average": avg_metrics
        }

        # Test pure BM25 search
        print("\nEvaluating pure BM25 search...")
        bm25_results = []
        for query_data in tqdm(self.test_queries):
            query = query_data["query"]
            relevant_docs = query_data["relevant_docs"]
            metrics = self.evaluate_search(
                self.bm25_search,
                query,
                relevant_docs,
                name="Pure BM25"
            )
            bm25_results.append(metrics)
        
        # Compute average metrics for BM25 search
        avg_metrics = {metric: np.mean([r[metric] for r in bm25_results]) 
                    for metric in bm25_results[0].keys()}
        self.results["pure_bm25"] = {
            "per_query": bm25_results,
            "average": avg_metrics
        }

        # Test hybrid with score method and different weights
        for semantic_weight in semantic_weights:
            bm25_weight = 1 - semantic_weight
            print(f"\nEvaluating hybrid score (semantic_weight={semantic_weight:.1f}, bm25_weight={bm25_weight:.1f})...")
            hybrid_score_results = []
            
            for query_data in tqdm(self.test_queries):
                query = query_data["query"]
                relevant_docs = query_data["relevant_docs"]
                metrics = self.evaluate_search(
                    lambda q, k: self.hybrid_search(q, k, semantic_weight=semantic_weight, 
                                                bm25_weight=bm25_weight, method='score'),
                    query,
                    relevant_docs,
                    name=f"Hybrid Score ({semantic_weight:.1f}, {bm25_weight:.1f})"
                )
                hybrid_score_results.append(metrics)
            
            # Compute average metrics
            avg_metrics = {metric: np.mean([r[metric] for r in hybrid_score_results]) 
                        for metric in hybrid_score_results[0].keys()}
            self.results[f"hybrid_score_{semantic_weight:.1f}"] = {
                "per_query": hybrid_score_results,
                "average": avg_metrics,
                "weights": {"semantic": semantic_weight, "bm25": bm25_weight}
            }

        # Test hybrid with RRF
        print("\nEvaluating hybrid RRF...")
        hybrid_rrf_results = []
        for query_data in tqdm(self.test_queries):
            query = query_data["query"]
            relevant_docs = query_data["relevant_docs"]
            metrics = self.evaluate_search(
                lambda q, k: self.hybrid_search(q, k, method='rrf', k_rrf=60),
                query,
                relevant_docs,
                name="Hybrid RRF"
            )
            hybrid_rrf_results.append(metrics)
        
        # Compute average metrics
        avg_metrics = {metric: np.mean([r[metric] for r in hybrid_rrf_results]) 
                    for metric in hybrid_rrf_results[0].keys()}
        self.results["hybrid_rrf"] = {
            "per_query": hybrid_rrf_results,
            "average": avg_metrics
        }

        # Print and plot results
        self.print_results()
        self.plot_results()
    
    def print_results(self):
        """Print results in a tabular format"""
        if not self.results:
            print("No benchmark results found. Please run the benchmark first.")
            return
        
        print("\n=== BENCHMARK RESULTS ===")
        
        # Get all metrics and algorithms
        metrics = list(next(iter(self.results.values()))["average"].keys())
        algorithms = list(self.results.keys())
        
        # Print average metrics
        print("\nAverage Metrics:")
        header = "Algorithm".ljust(20)
        for metric in metrics:
            header += metric.ljust(12)
        print(header)
        print("-" * 100)
        
        for algo in algorithms:
            row = algo.ljust(20)
            for metric in metrics:
                value = self.results[algo]["average"][metric]
                if metric == "query_time":
                    row += f"{value*1000:.2f}ms".ljust(12)  # Convert to milliseconds
                else:
                    row += f"{value:.4f}".ljust(12)
            print(row)
    
    def plot_results(self):
        """Plot benchmark results"""
        if not self.results:
            print("No benchmark results found. Please run the benchmark first.")
            return
        
        # Get data for plotting
        algorithms = list(self.results.keys())
        metrics = ["precision", "recall", "f1", "ndcg", "mrr"]
        
        metric_data = {
            metric: [self.results[algo]["average"][metric] for algo in algorithms]
            for metric in metrics
        }
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        # Plot metrics comparison
        for i, metric in enumerate(metrics):
            ax = axes[i]
            ax.bar(algorithms, metric_data[metric])
            ax.set_title(f"{metric.upper()}")
            ax.set_ylim(0, 1)
            ax.set_xticks(range(len(algorithms)))
            ax.set_xticklabels(algorithms, rotation=45, ha="right")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add values on top of bars
            for j, value in enumerate(metric_data[metric]):
                ax.text(j, value + 0.02, f"{value:.3f}", ha='center')
        
        # Plot query time
        query_times = [self.results[algo]["average"]["query_time"] * 1000 for algo in algorithms]  # Convert to ms
        ax = axes[5]
        ax.bar(algorithms, query_times)
        ax.set_title("Query Time (ms)")
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=45, ha="right")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for j, value in enumerate(query_times):
            ax.text(j, value + 0.5, f"{value:.2f}", ha='center')
        
        plt.tight_layout()
        plt.savefig("search_benchmark_results.png", dpi=300, bbox_inches="tight")
        print("\nResults plot saved as search_benchmark_results.png")
        
        # Plot hybrid weight optimization for score-based methods only
        hybrid_score_algos = [algo for algo in algorithms if algo.startswith("hybrid_score_")]
        if len(hybrid_score_algos) > 1:
            weights = [self.results[algo]["weights"]["semantic"] for algo in hybrid_score_algos]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for metric in metrics:
                values = [self.results[algo]["average"][metric] for algo in hybrid_score_algos]
                ax.plot(weights, values, marker='o', label=metric.upper())
            
            ax.set_xlabel("Semantic Weight")
            ax.set_ylabel("Score")
            ax.set_title("Effect of Semantic Weight on Search Quality")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig("hybrid_weight_optimization.png", dpi=300, bbox_inches="tight")
            print("Hybrid weight optimization plot saved as hybrid_weight_optimization.png")

def main():
    parser = argparse.ArgumentParser(description="Search Algorithm Benchmark Tool")
    parser.add_argument("--create", type=int, default=0, help="Create a new test dataset with N queries")
    parser.add_argument("--weights", type=str, default="0.5,0.6,0.7,0.8,0.9", help="Comma-separated list of semantic weights to test")
    args = parser.parse_args()
    
    benchmark = SearchBenchmark()
    benchmark.load_database_content()
    _ = benchmark.embedding_search("warm-up query", top_k=10)
    # Now measure a fresh call
    start = time.time()
    result = benchmark.embedding_search("warm-up query", top_k=10)
    elapsed_time = time.time() - start
    print(f"Pure embedding query time: {elapsed_time * 1000:.2f}ms")

    if args.create > 0:
        benchmark.create_test_dataset(args.create)
    
    weights = [float(w) for w in args.weights.split(",")]
    print(f"Testing with semantic weights: {weights}")
    
    benchmark.run_benchmark(semantic_weights=weights)

if __name__ == "__main__":
    main()