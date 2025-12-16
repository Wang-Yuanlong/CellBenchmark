import logging
import os

def setup_logging(output_dir: str):
    """Setup logging - only main process logs to console and file."""
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    
    log_level = logging.INFO if rank == 0 else logging.WARNING
    
    handlers = [logging.StreamHandler()]
    
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        handlers.append(logging.FileHandler(f"{output_dir}/train.log"))
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log_level,
        handlers=handlers,
    )
    
    return logging.getLogger(__name__)