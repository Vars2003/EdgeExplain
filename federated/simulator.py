import pandas as pd
import numpy as np
from typing import List

class FederatedSimulator:
    """
    Simulates federated client datasets by partitioning the processed active DataFrame.
    Consumes the session state data directly with zero duplicate reloading.
    """
    @staticmethod
    def partition_dataset(df: pd.DataFrame, num_clients: int = 3, strategy: str = "equal") -> List[pd.DataFrame]:
        if df is None or len(df) == 0:
            return []
            
        num_clients = max(1, num_clients)
        df_work = df.copy()
        
        clean_strategy = strategy.lower().strip()
        
        if clean_strategy == "random":
            # Shuffle rows randomly
            df_work = df_work.sample(frac=1.0, random_state=42).reset_index(drop=True)
        elif clean_strategy == "non-iid":
            # Sort by target column to partition non-uniformly (places similar classes together)
            target = df.columns[-1]
            df_work = df_work.sort_values(by=target).reset_index(drop=True)
        # "equal" strategy maintains original chronological / row sorting
        
        splits = np.array_split(df_work, num_clients)
        return [pd.DataFrame(s) for s in splits]
