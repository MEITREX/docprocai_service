from pgvector.psycopg import register_vector
import psycopg
import src.fileextractlib.LlamaRunner as LlamaRunner
import threading
import queue
from typing import Callable, Self


class DocProcAiService:
    def __init__(self):
        # TODO: Make db connection configurable
        self._db_conn: psycopg.Connection = psycopg.connect(
            "user=root password=root host=localhost port=5431 dbname=search-service",
            autocommit=True,
            row_factory=psycopg.rows.dict_row
        )

        # ensure pgvector extension is installed, we need it to store text embeddings
        self._db_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._db_conn)

        # db_conn.execute("DROP TABLE IF EXISTS documents")
        # db_conn.execute("DROP TABLE IF EXISTS videos")

        # ensure database tables exist
        self._db_conn.execute("CREATE TABLE IF NOT EXISTS documents "
                              + "(PRIMARY KEY(origin_file, page), "
                              + "text text, "
                              + "origin_file text, "
                              + "page int, "
                              + "embedding vector(1024))")
        self._db_conn.execute("CREATE TABLE IF NOT EXISTS videos "
                              + "(PRIMARY KEY(origin_file, start_time), "
                              + "screen_text text, "
                              + "transcript text, "
                              + "origin_file text, "
                              + "start_time int, "
                              + "embedding vector(1024))")

        # only load the llamaRunner the first time we actually need it, not now
        self._llamaRunner: LlamaRunner.LlamaRunner | None = None

        self._background_task_queue: queue.PriorityQueue[DocProcAiService.BackgroundTaskItem] = queue.PriorityQueue()

        self._keep_background_task_thread_alive: threading.Event = threading.Event()
        self._keep_background_task_thread_alive.set()

        self._background_task_thread: threading.Thread = threading.Thread(target=_background_task_runner)
        self._background_task_thread.start()

    def __del__(self):
        self._keep_background_task_thread_alive = False

    class BackgroundTaskItem:
        def __init__(self, task: Callable[[], None], priority: int):
            self.task: Callable[[], None] = task
            self.priority: int = priority

        def __lt__(self, other: Self):
            return self.priority < other.priority


def _background_task_runner(task_queue: queue.PriorityQueue[DocProcAiService.BackgroundTaskItem],
                            keep_alive: threading.Event):
    while keep_alive.is_set():
        try:
            background_task_item = task_queue.get(block=True, timeout=5)
        except queue.Empty:
            continue
        background_task_item.task()
        task_queue.task_done()
