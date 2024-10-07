from typing import Optional

import psycopg

from pgvector.psycopg import register_vector
from psycopg.types.enum import register_enum, EnumInfo

from persistence.entities import *


class DbConnector:
    def __init__(self, conn_info: str):
        self.db_connection: psycopg.Connection = psycopg.connect(
            conn_info,
            autocommit=True,
            row_factory=psycopg.rows.dict_row
        )

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
                  'CONTENT'
                );
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
            """)
        info = EnumInfo.fetch(self.db_connection, "ingestion_entity_type")
        register_enum(info, self.db_connection, IngestionEntityTypeDbType)

        self.db_connection.execute("""
                                   CREATE TABLE IF NOT EXISTS media_record_ingestion_states (
                                     id uuid PRIMARY KEY,
                                     entity_type ingestion_entity_type,
                                     state ingestion_state
                                   );
                                   """)

        self.db_connection.execute("""
                                   CREATE TABLE IF NOT EXISTS media_records (
                                     id uuid PRIMARY KEY,
                                     summary text[],
                                     vtt text,
                                     tags text[]
                                   );
                                   """)

        # ensure database tables exist
        # table which contains the sections of all documents including their text, page number, and text embedding
        self.db_connection.execute("""
                                   CREATE TABLE IF NOT EXISTS document_segments (
                                     id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                                     text text,
                                     media_record_id uuid,
                                     page int,
                                     thumbnail bytea,
                                     title text,
                                     embedding vector(1024)
                                   );
                                   """)
        # table which contains the sections of all videos including their screen text, transcript, start time, and text
        self.db_connection.execute("""
                                   CREATE TABLE IF NOT EXISTS video_segments (
                                     id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                                     screen_text text,
                                     transcript text,
                                     media_record_id uuid,
                                     start_time int,
                                     thumbnail bytea,
                                     title text,
                                     embedding vector(1024)
                                   );
                                   """)
        # table which contains links between segments of different media records
        # we can't use foreign keys here because the segments live in multiple tables
        self.db_connection.execute("""
                                   CREATE TABLE IF NOT EXISTS media_record_links (
                                     content_id uuid,
                                     segment1_id uuid,
                                     segment2_id uuid
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

    def insert_document_segment(self, text: str, media_record_id: UUID, page_index: int,
                                thumbnail: bytes, title: Optional[str], embedding: Tensor) -> None:
        self.db_connection.execute(
            query="""
                  INSERT INTO document_segments (text, media_record_id, page, thumbnail, title, embedding) 
                  VALUES (%s, %s, %s, %s, %s, %s)
                  """,
            params=(text, media_record_id, page_index, thumbnail, title, embedding))

    def insert_video_segment(self, screen_text: str, transcript: str, media_record_id: UUID, start_time: int,
                             thumbnail: bytes, title: str, embedding: Tensor) -> None:
        self.db_connection.execute(
            query="""
                  INSERT INTO video_segments (
                    screen_text,
                    transcript,
                    media_record_id,
                    start_time,
                    thumbnail,
                    title,
                    embedding
                  )
                  VALUES (%s, %s, %s, %s, %s, %s, %s)
                  """,
            params=(screen_text, transcript, media_record_id, start_time, thumbnail, title, embedding))

    def insert_media_record_segment_link(self, content_id: UUID, segment1_id: UUID, segment2_id: UUID) -> None:
        self.db_connection.execute(
            query="""
                  INSERT INTO media_record_links (content_id, segment1_id, segment2_id)
                  VALUES (%s, %s, %s)
                  """,
            params=(content_id, segment1_id, segment2_id))

    def upsert_media_record(self, id: UUID, summary: list[str], vtt: Optional[str]):
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

    def update_media_record_tags(self, id: UUID, tags: list[str]):
        self.db_connection.execute(
            """
                UPDATE media_records
                SET tags = (%(tags)s)
                WHERE id = (%(id)s)
            """,
            {'tags': tags, 'id': id})

    def delete_media_record_segment_links_by_segment_ids(self, segment_ids: list[UUID]) -> list[MediaRecordSegmentLinkEntity]:
        query_result = self.db_connection.execute(
            """
                  DELETE FROM media_record_links 
                  WHERE segment1_id = ANY(%(segment_ids)s) OR segment2_id = ANY(%(segment_ids)s)
                  RETURNING *
                  """,
            {"segment_ids": segment_ids}).fetchall()

        return [DbConnector.__media_record_segment_link_query_result_to_object(x) for x in query_result]

    def delete_media_record_segment_links_by_content_ids(self, content_ids: list[UUID]) \
            -> list[MediaRecordSegmentLinkEntity]:
        query_result = self.db_connection.execute(
            """
            DELETE FROM media_record_links
            WHERE content_id = ANY(%(content_ids)s)
            RETURNING *;
            """, {"content_ids": content_ids}).fetchall()
        return [DbConnector.__media_record_segment_link_query_result_to_object(x) for x in query_result]

    def delete_document_segments_by_media_record_id(self, media_record_ids: list[UUID]) -> list[DocumentSegmentEntity]:
        query_results = self.db_connection.execute(
            """
                  DELETE FROM document_segments
                  WHERE media_record_id = ANY(%(mediaRecordIds)s)
                  RETURNING *
                  """,
            {"mediaRecordIds": media_record_ids})

        return [DbConnector.__document_segment_query_result_to_object(x) for x in query_results]

    def delete_video_segments_by_media_record_id(self, media_record_ids: list[UUID]) -> list[VideoSegmentEntity]:
        query_results = self.db_connection.execute(
            """
                  DELETE FROM video_segments
                  WHERE media_record_id = ANY(%(mediaRecordIds)s)
                  RETURNING *
                  """,
            {"mediaRecordIds": media_record_ids})

        return [DbConnector.__video_segment_query_result_to_object(x) for x in query_results]

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

    def get_segment_links_by_content_id(self, content_id: UUID) -> list[MediaRecordSegmentLinkEntity]:
        result = self.db_connection.execute("""
                    SELECT
                        segment1_id,
                        segment2_id,
                        content_id
                    FROM media_record_links WHERE content_id = %s
                    """, (content_id,)).fetchall()

        return [DbConnector.__media_record_segment_link_query_result_to_object(x) for x in result]

    def does_segment_link_exist(self, segment1_id: UUID, segment2_id: UUID, content_id: UUID = None) -> bool:
        if content_id is None:
            result = self.db_connection.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM media_record_links
                            WHERE (segment1_id = %s AND segment2_id = %s)
                            OR (segment1_id = %s AND segment2_id = %s)
                        )
                        """, (segment1_id, segment2_id, segment2_id, segment1_id)).fetchone()
        else:
            result = self.db_connection.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM media_record_links
                            WHERE content_id = %s AND (
                                (segment1_id = %s AND segment2_id = %s)
                                OR (segment1_id = %s AND segment2_id = %s)
                            )
                        )
                        """, (content_id, segment1_id, segment2_id, segment2_id, segment1_id)).fetchone()

        return result["exists"]

    def get_top_record_segments_by_embedding_distance(self, query_embedding: Tensor,
                                                      count: int,
                                                      media_record_id_blacklist: list[UUID],
                                                      media_record_id_whitelist: list[UUID]) \
            -> list[SemanticSearchResultEntity]:
        # sql query to get the closest embeddings to the query embedding, both from the video and document tables
        query = """
                    WITH document_results AS (
                        SELECT
                            id,
                            media_record_id,
                            'document' AS source,
                            page,
                            NULL::integer AS "start_time",
                            text,
                            NULL::text AS "screen_text",
                            NULL::text AS transcript,
                            title,
                            thumbnail,
                            embedding,
                            embedding <=> %(query_embedding)s AS score
                        FROM document_segments
                        WHERE media_record_id = ANY(%(mediaRecordWhitelist)s) AND NOT media_record_id = ANY(%(mediaRecordBlacklist)s)
                    ),
                    video_results AS (
                        SELECT 
                            id,
                            media_record_id,
                            'video' AS source,
                            NULL::integer AS page,
                            start_time,
                            NULL::text AS text,
                            screen_text,
                            transcript,
                            title,
                            thumbnail,
                            embedding,
                            embedding <=> %(query_embedding)s AS score
                        FROM video_segments
                        WHERE media_record_id = ANY(%(mediaRecordWhitelist)s) AND NOT media_record_id = ANY(%(mediaRecordBlacklist)s)
                    ),
                    results AS (
                        SELECT * FROM document_results
                        UNION ALL
                        SELECT * FROM video_results
                    )
                    SELECT * FROM results ORDER BY score LIMIT %(count)s
                """

        query_results = self.db_connection.execute(query, {
            "query_embedding": query_embedding,
            "count": count,
            "mediaRecordBlacklist": media_record_id_blacklist,
            "mediaRecordWhitelist": media_record_id_whitelist
        }).fetchall()

        return [SemanticSearchResultEntity(
            x["score"],
            DbConnector.__media_record_segment_query_result_to_object(x)
        ) for x in query_results]

    def get_record_segments_by_media_record_ids(self, media_record_ids: list[UUID]) \
            -> list[DocumentSegmentEntity | VideoSegmentEntity]:
        query = """
                WITH document_results AS (
                    SELECT
                        id,
                        media_record_id,
                        'document' AS source,
                        page,
                        NULL::integer AS start_time,
                        NULL::text AS screen_text,
                        NULL::text AS transcript,
                        text,
                        thumbnail,
                        title,
                        embedding
                    FROM document_segments
                    WHERE media_record_id = ANY(%(mediaRecordIds)s)
                ),
                video_results AS (
                    SELECT 
                        id,
                        media_record_id,
                        'video' AS source,
                        NULL::integer AS page,
                        start_time,
                        screen_text,
                        transcript,
                        NULL::text AS text,
                        thumbnail,
                        title,
                        embedding
                    FROM video_segments
                    WHERE media_record_id = ANY(%(mediaRecordIds)s)
                ),
                results AS (
                    SELECT * FROM document_results
                    UNION ALL
                    SELECT * FROM video_results
                )
                SELECT * FROM results
                """
        return self.__get_record_segments_with_query(query, {"mediaRecordIds": media_record_ids})

    def get_all_media_records(self):
        cursor = self.db_connection.cursor()
        cursor.execute(
            "SELECT * FROM media_records"
        )
        return cursor.fetchall()

    def get_all_record_segments(self) \
            -> list[DocumentSegmentEntity | VideoSegmentEntity]:
        query = """
                WITH document_results AS (
                    SELECT
                        id,
                        media_record_id,
                        'document' AS source,
                        page,
                        NULL::integer AS start_time,
                        NULL::text AS screen_text,
                        NULL::text AS transcript,
                        text,
                        thumbnail,
                        title,
                        embedding
                    FROM document_segments
                ),
                video_results AS (
                    SELECT 
                        id,
                        media_record_id,
                        'video' AS source,
                        NULL::integer AS page,
                        start_time,
                        screen_text,
                        transcript,
                        NULL::text AS text,
                        thumbnail,
                        title,
                        embedding
                    FROM video_segments
                ),
                results AS (
                    SELECT * FROM document_results
                    UNION ALL
                    SELECT * FROM video_results
                )
                SELECT * FROM results
                """
        return self.__get_record_segments_with_query(query, {})

    def get_record_segments_by_ids(self, segment_ids: list[UUID]) -> list[DocumentSegmentEntity | VideoSegmentEntity]:
        query = """
                WITH document_results AS (
                    SELECT
                        id,
                        media_record_id,
                        'document' AS source,
                        page,
                        NULL::integer AS start_time,
                        text,
                        NULL::text AS screen_text,
                        NULL::text AS transcript,
                        thumbnail,
                        title,
                        embedding
                    FROM document_segments
                    WHERE id = ANY(%(segmentIds)s)
                ),
                video_results AS (
                    SELECT 
                        id,
                        media_record_id,
                        'video' AS source,
                        NULL::integer AS page,
                        start_time,
                        NULL::text AS text,
                        screen_text,
                        transcript,
                        thumbnail,
                        title,
                        embedding
                    FROM video_segments
                    WHERE id = ANY(%(segmentIds)s)
                ),
                results AS (
                    SELECT * FROM document_results
                    UNION ALL
                    SELECT * FROM video_results
                )
                SELECT * FROM results;
                """
        return self.__get_record_segments_with_query(query, {"segmentIds": segment_ids})

    def __get_record_segments_with_query(self, query: str, params: dict) \
            -> list[DocumentSegmentEntity | VideoSegmentEntity]:
        query_results = self.db_connection.execute(query=query, params=params).fetchall()

        entities = []
        for x in query_results:
            entity = DbConnector.__media_record_segment_query_result_to_object(x)
            entities.append(entity)

        return entities

    @staticmethod
    def __media_record_segment_query_result_to_object(query_result) -> DocumentSegmentEntity | VideoSegmentEntity:
        if query_result["source"] == "document":
            return DbConnector.__document_segment_query_result_to_object(query_result)
        elif query_result["source"] == "video":
            return DbConnector.__video_segment_query_result_to_object(query_result)

    @staticmethod
    def __document_segment_query_result_to_object(query_result) -> DocumentSegmentEntity:
        return DocumentSegmentEntity(query_result["id"], query_result["media_record_id"], query_result["page"],
                                     query_result["text"], bytes(query_result["thumbnail"]), query_result["title"],
                                     query_result["embedding"])

    @staticmethod
    def __video_segment_query_result_to_object(query_result) -> VideoSegmentEntity:
        return VideoSegmentEntity(query_result["id"], query_result["media_record_id"], query_result["start_time"],
                                  query_result["transcript"], query_result["screen_text"],
                                  bytes(query_result["thumbnail"]), query_result["title"], query_result["embedding"])

    @staticmethod
    def __media_record_segment_link_query_result_to_object(query_result) -> MediaRecordSegmentLinkEntity:
        return MediaRecordSegmentLinkEntity(query_result["content_id"], query_result["segment1_id"],
                                            query_result["segment2_id"])

    def __del__(self):
        if self.db_connection is not None:
            self.db_connection.close()
