from uuid import UUID

import psycopg
from pgvector.psycopg import register_vector


class AssessmentInfoDbConnector:
    def __init__(self, db_connection: psycopg.Connection):
        self.db_connection = db_connection

        # ensure pgvector extension is installed, we need it to store text embeddings
        self.db_connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.db_connection)

        self.db_connection.execute(
            """
            CREATE TABLE IF NOT EXISTS assessments (
              id uuid PRIMARY KEY,
              tags text[]
            );
            """)

    def add_assessment_info(self, id: UUID):
        self.db_connection.execute(
            query="""
                  INSERT INTO assessments (id)
                  VALUES (%s)
                  """,
            params=(id)
        )

    def get_assessment_tags_by_id(self, assesment_id) -> list[str]:
        query_result = self.db_connection.execute(
            "SELECT tags FROM assessments WHERE id = %s",
            (assesment_id,)).fetchone()

        if query_result is None:
            return []

        return query_result["tags"]

    def get_all_assessments(self):
        cursor = self.db_connection.cursor()
        cursor.execute(
            "SELECT * FROM assessments"
        )
        return cursor.fetchall()

    def update_assessment_tags(self, id: UUID, tags: list[str]):
        self.db_connection.execute(
            """
                UPDATE assessments
                SET tags = (%(tags)s)
                WHERE id = (%(id)s)
            """,
            {'tags': tags, 'id': id})
