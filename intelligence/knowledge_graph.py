from typing import Dict, Any, List, Optional
import pandas as pd
from utils.logger import get_logger

logger = get_logger("intelligence.knowledge_graph")

class KnowledgeGraph:
    """
    Standardized relational graph data structure mapping:
    Dataset ➔ Features ➔ Dependencies ➔ Recommendations ➔ Algorithms ➔ Insights.
    Exposes graph query APIs.
    """

    def __init__(self):
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        logger.info("Empty Knowledge Graph initialized.")

    def add_node(self, node_id: str, node_type: str, label: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        # Prevent duplicates
        if any(n["id"] == node_id for n in self.nodes):
            return
        self.nodes.append({
            "id": node_id,
            "type": node_type,
            "label": label,
            "metadata": metadata or {}
        })

    def add_edge(self, source: str, target: str, relation: str, confidence: int = 100) -> None:
        # Prevent duplicates
        if any(e["source"] == source and e["target"] == target and e["relation"] == relation for e in self.edges):
            return
        self.edges.append({
            "source": source,
            "target": target,
            "relation": relation,
            "confidence": confidence
        })

    def build_graph(self, context: Dict[str, Any], semantic_types: Dict[str, Dict[str, Any]], metrics: Dict[str, Any], recs: List[Dict[str, Any]], algo_recs: List[Dict[str, Any]]) -> None:
        """
        Populates nodes and edges from computed profile metadata.
        """
        logger.info("Building knowledge graph...")
        self.nodes.clear()
        self.edges.clear()
        
        # 1. Dataset Node
        ds_id = f"ds_{context['dataset_name']}"
        self.add_node(
            node_id=ds_id,
            node_type="dataset",
            label=context["dataset_name"],
            metadata={
                "row_count": context["row_count"],
                "column_count": context["column_count"],
                "domain": context["domain"],
                "dataset_type": context["dataset_type"],
                "quality_score": context["quality_overview"]["quality_score"]
            }
        )
        
        # 2. Features Nodes
        for col, info in semantic_types.items():
            feat_id = f"feat_{col}"
            self.add_node(
                node_id=feat_id,
                node_type="feature",
                label=col,
                metadata={
                    "semantic_type": info["type"],
                    "pandas_type": str(info["evidence"][0]) if info["evidence"] else "unknown"
                }
            )
            # Edge: Dataset -> Feature
            self.add_edge(ds_id, feat_id, "has_feature", 100)
            
        # 3. Dependency Nodes & Edges
        dep_data = metrics.get("dependency", {})
        if dep_data:
            cols = dep_data.get("columns", [])
            matrix = dep_data.get("matrix", [])
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    w = matrix[i][j]
                    if w > 0.3: # Threshold check
                        dep_id = f"dep_{cols[i]}_{cols[j]}"
                        self.add_node(
                            node_id=dep_id,
                            node_type="dependency",
                            label=f"{cols[i]} <-> {cols[j]} Relationship",
                            metadata={"weight": float(w)}
                        )
                        # Edges
                        self.add_edge(f"feat_{cols[i]}", dep_id, "relates_to", int(w * 100))
                        self.add_edge(f"feat_{cols[j]}", dep_id, "relates_to", int(w * 100))

        # 4. Recommendation Nodes & Edges
        for r in recs:
            rec_id = f"rec_{r['type']}_{'_'.join(r['features'])}"
            self.add_node(
                node_id=rec_id,
                node_type="recommendation",
                label=f"Recommend {r['type']}",
                metadata={
                    "fix": r["suggested_fix"],
                    "reason": r["reason"],
                    "alternatives": r["alternatives"]
                }
            )
            
            # Edges: Features -> Recommendation
            for feat in r["features"]:
                self.add_edge(f"feat_{feat}", rec_id, "requires", r["confidence"])
                
            # Edge: Dataset -> Recommendation
            if not r["features"]:
                self.add_edge(ds_id, rec_id, "requires", r["confidence"])

        # 5. Algorithm Nodes & Edges
        for a in algo_recs:
            algo_id = f"algo_{a['algorithm'].replace(' ', '_')}"
            self.add_node(
                node_id=algo_id,
                node_type="algorithm",
                label=a["algorithm"],
                metadata={
                    "compatibility_score": a["compatibility_score"],
                    "pros": a["pros"],
                    "cons": a["cons"]
                }
            )
            # Edge: Dataset -> Algorithm
            self.add_edge(ds_id, algo_id, "compatible_with", a["compatibility_score"])
            
        logger.info(f"Knowledge Graph construction complete. Nodes: {len(self.nodes)}, Edges: {len(self.edges)}")

    # --- Query APIs ---

    def find_feature(self, col_name: str) -> Dict[str, Any]:
        """
        Returns feature details and all adjacent nodes/edges.
        """
        feat_id = f"feat_{col_name}"
        node = next((n for n in self.nodes if n["id"] == feat_id), {})
        connected_edges = [e for e in self.edges if e["source"] == feat_id or e["target"] == feat_id]
        return {
            "node": node,
            "connections": connected_edges
        }

    def find_dependency(self, col_a: str, col_b: str) -> Optional[Dict[str, Any]]:
        """
        Returns relationship weights between two columns.
        """
        dep_id_1 = f"dep_{col_a}_{col_b}"
        dep_id_2 = f"dep_{col_b}_{col_a}"
        node = next((n for n in self.nodes if n["id"] in [dep_id_1, dep_id_2]), None)
        return node

    def find_algorithm(self, algo_name: str) -> Dict[str, Any]:
        """
        Retrieves matching algorithm compatibility.
        """
        algo_id = f"algo_{algo_name.replace(' ', '_')}"
        return next((n for n in self.nodes if n["id"] == algo_id), {})

    def find_recommendation(self, rec_type: str) -> List[Dict[str, Any]]:
        """
        Retrieves all recommendations matching the requested preprocessing type.
        """
        return [n for n in self.nodes if n["type"] == "recommendation" and rec_type.lower() in n["id"].lower()]

    def find_summary(self) -> Dict[str, Any]:
        """
        Retrieves high-level summary info from the dataset node.
        """
        return next((n for n in self.nodes if n["type"] == "dataset"), {})
