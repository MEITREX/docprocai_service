from uuid import UUID

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.enum import register_enum, EnumInfo

from persistence.entities import IngestionStateDbType, IngestionEntityTypeDbType, EntityIngestionInfoEntity


class IngestionStateDbConnector:
    def __init__(self, db_connection: psycopg.Connection):
        self.db_connection = db_connection

        # ensure pgvector extension is installed, we need it to store text embeddings
        self.db_connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.db_connection)

        self.db_connection.execute(
            """
            DO $$ BEGIN
                CREATE TYPE ingestion_state AS ENUM (
                  'ENQUEUED',
                  'PROCESSING',
                  'DONE'
                );
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
            """)
        info = EnumInfo.fetch(self.db_connection, "ingestion_state")
        register_enum(info, self.db_connection, IngestionStateDbType)

        self.db_connection.execute(
            """
            DO $$ BEGIN
                CREATE TYPE ingestion_entity_type AS ENUM (
                  'MEDIA_RECORD',
                  'MEDIA_CONTENT',
                  'ASSESSMENT'
                );
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
            """)
        info = EnumInfo.fetch(self.db_connection, "ingestion_entity_type")
        register_enum(info, self.db_connection, IngestionEntityTypeDbType)

        self.db_connection.execute(
            """
            CREATE TABLE IF NOT EXISTS media_record_ingestion_states (
              id uuid PRIMARY KEY,
              entity_type ingestion_entity_type,
              state ingestion_state
            );
            """)

    def upsert_entity_ingestion_info(self,
                                     entity_id: UUID,
                                     ingestion_entity_type: IngestionEntityTypeDbType,
                                     ingestion_state: IngestionStateDbType) -> None:
        self.db_connection.execute(
            """
            INSERT INTO media_record_ingestion_states (id, entity_type, state)
            VALUES (%(id)s, %(entity_type)s, %(state)s)
            ON CONFLICT(id)
            DO UPDATE SET
              entity_type = EXCLUDED.entity_type,
              state = EXCLUDED.state;
            """,
            params={
                "id": entity_id,
                "entity_type": ingestion_entity_type,
                "state": ingestion_state
            })

    def get_entities_ingestion_info(self, entity_ids: list[UUID]) -> list[EntityIngestionInfoEntity]:
        """
        Returns the ingestion states stored in the DB for the entities with the given IDs. Returned entity list may
        not be in the same order as the passed IDs. If no entity exists in the DB with the given ID, it is excluded
        from the returned list.
        """
        query_results = self.db_connection.execute(
            """
            SELECT id, state, entity_type
            FROM media_record_ingestion_states
            WHERE id = ANY(%s);
            """,
            params=(entity_ids,)).fetchall()

        return [EntityIngestionInfoEntity(
            entity_id=result["id"],
            entity_type=result["entity_type"],
            ingestion_state=result["state"]
        ) for result in query_results]

    def get_enqueued_or_processing_ingestion_entities(self) \
            -> list[tuple[UUID, IngestionEntityTypeDbType, IngestionStateDbType]]:
        query_results = self.db_connection.execute(
            """
            SELECT id, state, entity_type
            FROM media_record_ingestion_states
            WHERE state IN ('ENQUEUED', 'PROCESSING');
            """).fetchall()
        return [(x["id"], x["entity_type"], x["state"]) for x in query_results]

    def delete_ingestion_state(self, id: UUID) -> None:
        self.db_connection.execute(
            """
            DELETE FROM media_record_ingestion_states WHERE id = ANY(%(id)s);
            """,
            {'id': id}
        )