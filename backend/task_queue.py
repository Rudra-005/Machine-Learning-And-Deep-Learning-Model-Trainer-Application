"""
Task queue for handling async training jobs
"""
import threading
from queue import Queue
from app.utils.logger import logger

class TaskQueue:
    """Simple task queue for background processing"""
    
    _queue = Queue()
    _running = False
    _worker_thread = None
    
    @staticmethod
    def start():
        """Start task queue worker"""
        if not TaskQueue._running:
            TaskQueue._running = True
            TaskQueue._worker_thread = threading.Thread(
                target=TaskQueue._worker, daemon=True
            )
            TaskQueue._worker_thread.start()
            logger.info("Task queue started")
    
    @staticmethod
    def stop():
        """Stop task queue worker"""
        TaskQueue._running = False
        if TaskQueue._worker_thread:
            TaskQueue._worker_thread.join(timeout=5)
        logger.info("Task queue stopped")
    
    @staticmethod
    def enqueue(task_func, *args, **kwargs):
        """
        Enqueue task for processing
        
        Args:
            task_func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        TaskQueue._queue.put((task_func, args, kwargs))
        logger.debug(f"Task enqueued: {task_func.__name__}")
    
    @staticmethod
    def _worker():
        """Worker thread that processes queued tasks"""
        while TaskQueue._running:
            try:
                task_func, args, kwargs = TaskQueue._queue.get(timeout=1)
                logger.info(f"Executing task: {task_func.__name__}")
                task_func(*args, **kwargs)
            except:
                pass  # Queue.Empty exception
