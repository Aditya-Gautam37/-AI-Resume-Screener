"""
Batch Processor Module
Handles parallel processing of multiple resumes
"""

from multiprocessing import Pool, cpu_count
from typing import List, Callable, Any
from tqdm import tqdm
import time
from loguru import logger

class BatchProcessor:
    """Process multiple resumes in parallel for speed"""
    
    def __init__(self, num_workers: int = None):
        """
        Initialize batch processor
        
        Args:
            num_workers: Number of parallel workers (default: CPU count)
        """
        self.num_workers = num_workers or cpu_count()
        logger.info(f"BatchProcessor initialized with {self.num_workers} workers")
    
    def process_batch(self, items: List[Any], process_func: Callable, 
                      chunk_size: int = 10, desc: str = "Processing") -> List[Any]:
        """
        Process items in parallel batches
        
        Args:
            items: List of items to process
            process_func: Function to apply to each item
            chunk_size: Size of chunks for workers
            desc: Description for progress bar
            
        Returns:
            List of processed results
        """
        if not items:
            logger.warning("No items to process")
            return []
        
        logger.info(f"Processing {len(items)} items with {self.num_workers} workers")
        start_time = time.time()
        
        with Pool(processes=self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, items, chunksize=chunk_size),
                total=len(items),
                desc=desc,
                unit="items"
            ))
        
        elapsed = time.time() - start_time
        logger.info(f"Processed {len(items)} items in {elapsed:.2f} seconds")
        logger.info(f"Speed: {len(items)/elapsed:.2f} items/second")
        
        return results
    
    def process_in_chunks(self, items: List[Any], process_func: Callable, 
                          chunk_size: int = 100) -> List[Any]:
        """
        Process in sequential chunks (for memory-intensive operations)
        
        Args:
            items: List of items to process
            process_func: Function to apply to each chunk
            chunk_size: Size of each chunk
            
        Returns:
            List of processed results
        """
        results = []
        
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(items)-1)//chunk_size + 1}")
            chunk_results = process_func(chunk)
            results.extend(chunk_results)
        
        return results