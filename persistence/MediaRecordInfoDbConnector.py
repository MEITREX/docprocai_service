from typing import Optional
from uuid import UUID

import psycopg
from pgvector.psycopg import register_vector


class MediaRecordInfoDbConnector:
    def __init__(self, db_connection: psycopg.Connection):
        self.db_connection = db_connection

        # ensure pgvector extension is installed, we need it to store text embeddings
        self.db_connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.db_connection)

        self.db_connection.execute(
            """
            CREATE TABLE IF NOT EXISTS media_records (
              id uuid PRIMARY KEY,
              summary text[],
              vtt text,
              tags text[]
            );
            """)

    def upsert_media_record_info(self, id: UUID, summary: list[str], vtt: Optional[str]):
        self.db_connection.execute(
            query="""
                  INSERT INTO media_records (id, summary, vtt)
                  VALUES (%s, %s, %s)
                  ON CONFLICT (id)
                  DO UPDATE SET
                      summary = EXCLUDED.summary,
                      vtt = EXCLUDED.vtt;
                  """,
            params=(id, summary, vtt)
        )

    def get_media_record_summary_by_media_record_id(self, media_record_id) -> list[str]:
        query_result = self.db_connection.execute(
            "SELECT summary FROM media_records WHERE media_record_id = %s",
            (media_record_id,)).fetchone()

        if query_result is None:
            return []

        return query_result["summary"]

    def get_video_captions_by_media_record_id(self, media_record_id: UUID) -> str | None:
        query_result = self.db_connection.execute(
            "SELECT vtt FROM media_records WHERE id = %s",
            (media_record_id,)).fetchone()

        if query_result is None:
            return None

        return query_result["vtt"]

    def get_media_record_tags_by_media_record_id(self, media_record_id) -> list[str]:
        query_result = self.db_connection.execute(
            "SELECT tags FROM media_records WHERE media_record_id = %s",
            (media_record_id,)).fetchone()

        if query_result is None:
            return []

        return query_result["tags"]

    def get_all_media_records(self):
        cursor = self.db_connection.cursor()
        cursor.execute(
            "SELECT * FROM media_records"
        )
        return cursor.fetchall()

    def update_media_record_tags(self, id: UUID, tags: list[str]):
        self.db_connection.execute(
            """
                UPDATE media_records
                SET tags = (%(tags)s)
                WHERE id = (%(id)s)
            """,
            {'tags': tags, 'id': id})
