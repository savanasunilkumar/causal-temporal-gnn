"""Universal data processor for recommendation systems."""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List


class DataProcessor:
    """
    Universal data processor that automatically detects data format and schema.

    Supports CSV, JSON, and Parquet formats with automatic column type detection.
    """

    def __init__(self, config):
        """
        Initialize the data processor.

        Args:
            config: Configuration object with processing parameters
        """
        self.config = config
        self.metadata = {}
        self._column_patterns = {
            'user': ['user', 'userid', 'user_id', 'uid', 'customer', 'customerid', 'customer_id'],
            'item': ['item', 'itemid', 'item_id', 'product', 'productid', 'product_id',
                     'movie', 'movieid', 'movie_id', 'book', 'bookid', 'book_id'],
            'rating': ['rating', 'score', 'stars', 'value', 'preference'],
            'timestamp': ['timestamp', 'time', 'date', 'datetime', 'ts', 'created_at', 'created'],
            'text': ['title', 'name', 'description', 'review', 'comment', 'text', 'content'],
            'category': ['category', 'genre', 'type', 'class', 'tag', 'label'],
        }

    def process_data(self, data_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load and process data from a file.

        Args:
            data_path: Path to the data file (CSV, JSON, or Parquet)

        Returns:
            Tuple of (processed DataFrame, detected schema)
        """
        # Load data based on file extension
        data = self._load_data(data_path)

        # Detect schema
        schema = self._detect_schema(data)

        # Update metadata
        self._update_metadata(data, schema)

        return data, schema

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from file based on extension."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        ext = os.path.splitext(data_path)[1].lower()

        if ext == '.csv':
            data = pd.read_csv(data_path)
        elif ext == '.json':
            data = pd.read_json(data_path)
        elif ext == '.parquet':
            data = pd.read_parquet(data_path)
        elif ext == '.tsv':
            data = pd.read_csv(data_path, sep='\t')
        else:
            # Try CSV as default
            try:
                data = pd.read_csv(data_path)
            except Exception:
                raise ValueError(f"Unsupported file format: {ext}")

        print(f"Loaded {len(data)} rows from {data_path}")
        return data

    def _detect_schema(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Automatically detect column types based on names and content.

        Returns:
            Dictionary mapping column types to column names
        """
        schema = {
            'user_columns': [],
            'item_columns': [],
            'interaction_columns': [],
            'temporal_columns': [],
            'text_columns': [],
            'numeric_columns': [],
            'categorical_columns': [],
            'image_columns': [],
            'context_columns': [],
        }

        for col in data.columns:
            col_lower = col.lower().replace('_', '').replace('-', '')
            dtype = data[col].dtype

            # Check for user columns
            if any(pattern in col_lower for pattern in self._column_patterns['user']):
                schema['user_columns'].append(col)
            # Check for item columns
            elif any(pattern in col_lower for pattern in self._column_patterns['item']):
                schema['item_columns'].append(col)
            # Check for rating/interaction columns
            elif any(pattern in col_lower for pattern in self._column_patterns['rating']):
                schema['interaction_columns'].append(col)
            # Check for timestamp columns
            elif any(pattern in col_lower for pattern in self._column_patterns['timestamp']):
                schema['temporal_columns'].append(col)
            # Check for text columns
            elif any(pattern in col_lower for pattern in self._column_patterns['text']):
                schema['text_columns'].append(col)
            # Check for categorical columns
            elif any(pattern in col_lower for pattern in self._column_patterns['category']):
                schema['categorical_columns'].append(col)
            # Check data type for remaining columns
            elif dtype == 'object':
                # Check if it looks like a file path (image)
                sample = data[col].dropna().head(10)
                if any(str(s).endswith(('.jpg', '.png', '.jpeg', '.gif')) for s in sample):
                    schema['image_columns'].append(col)
                elif data[col].nunique() < len(data) * 0.5:  # Likely categorical
                    schema['categorical_columns'].append(col)
                else:
                    schema['text_columns'].append(col)
            elif np.issubdtype(dtype, np.number):
                schema['numeric_columns'].append(col)

        # Ensure we have at least user and item columns
        if not schema['user_columns'] and not schema['item_columns']:
            # Try to infer from first two columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                schema['user_columns'] = [numeric_cols[0]]
                schema['item_columns'] = [numeric_cols[1]]
                if len(numeric_cols) >= 3:
                    schema['interaction_columns'] = [numeric_cols[2]]
                if len(numeric_cols) >= 4:
                    schema['temporal_columns'] = [numeric_cols[3]]

        print(f"Detected schema: {schema}")
        return schema

    def _update_metadata(self, data: pd.DataFrame, schema: Dict[str, List[str]]):
        """Update metadata based on processed data."""
        user_col = schema['user_columns'][0] if schema['user_columns'] else None
        item_col = schema['item_columns'][0] if schema['item_columns'] else None

        self.metadata = {
            'num_rows': len(data),
            'num_columns': len(data.columns),
            'columns': list(data.columns),
            'num_users': data[user_col].nunique() if user_col else 0,
            'num_items': data[item_col].nunique() if item_col else 0,
            'user_column': user_col,
            'item_column': item_col,
            'has_ratings': len(schema['interaction_columns']) > 0,
            'has_timestamps': len(schema['temporal_columns']) > 0,
            'has_text': len(schema['text_columns']) > 0,
            'has_images': len(schema['image_columns']) > 0,
            'sparsity': self._compute_sparsity(data, user_col, item_col),
        }

    def _compute_sparsity(self, data: pd.DataFrame, user_col: Optional[str],
                          item_col: Optional[str]) -> float:
        """Compute matrix sparsity."""
        if user_col is None or item_col is None:
            return 0.0

        num_users = data[user_col].nunique()
        num_items = data[item_col].nunique()
        num_interactions = len(data)

        if num_users == 0 or num_items == 0:
            return 0.0

        density = num_interactions / (num_users * num_items)
        sparsity = 1 - density
        return sparsity * 100  # As percentage

    def filter_data(self, data: pd.DataFrame, schema: Dict[str, List[str]],
                    min_user_interactions: int = 5,
                    min_item_interactions: int = 5) -> pd.DataFrame:
        """
        Filter data to remove users and items with too few interactions.

        Args:
            data: DataFrame to filter
            schema: Detected schema
            min_user_interactions: Minimum interactions per user
            min_item_interactions: Minimum interactions per item

        Returns:
            Filtered DataFrame
        """
        user_col = schema['user_columns'][0] if schema['user_columns'] else None
        item_col = schema['item_columns'][0] if schema['item_columns'] else None

        if user_col is None or item_col is None:
            return data

        # Iteratively filter until convergence
        prev_len = len(data) + 1
        while len(data) < prev_len:
            prev_len = len(data)

            # Filter users
            user_counts = data[user_col].value_counts()
            active_users = user_counts[user_counts >= min_user_interactions].index
            data = data[data[user_col].isin(active_users)]

            # Filter items
            item_counts = data[item_col].value_counts()
            active_items = item_counts[item_counts >= min_item_interactions].index
            data = data[data[item_col].isin(active_items)]

        return data

    def create_id_mappings(self, data: pd.DataFrame, schema: Dict[str, List[str]]) -> Tuple[Dict, Dict]:
        """
        Create contiguous ID mappings for users and items.

        Args:
            data: DataFrame with user and item columns
            schema: Detected schema

        Returns:
            Tuple of (user_id_map, item_id_map)
        """
        user_col = schema['user_columns'][0] if schema['user_columns'] else None
        item_col = schema['item_columns'][0] if schema['item_columns'] else None

        user_id_map = {}
        item_id_map = {}

        if user_col:
            unique_users = sorted(data[user_col].unique())
            user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}

        if item_col:
            unique_items = sorted(data[item_col].unique())
            item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}

        return user_id_map, item_id_map
