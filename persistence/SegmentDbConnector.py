from typing import Optional

import psycopg

from pgvector.psycopg import register_vector

from persistence.entities import *


class SegmentDbConnector:
    def __init__(self, db_connection: psycopg.Connection):
        self.db_connection: psycopg.Connection = db_connection

        # ensure pgvector extension is installed, we need it to store text embeddings
        self.db_connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.db_connection)

        # ensure database tables exist
        # table which contains the sections of all documents including their text, page number, and text embedding
        self.db_connection.execute(
            """
            CREATE TABLE IF NOT EXISTS document_segments (
              id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
              media_record_id uuid,
              text text,
              page int,
              thumbnail bytea,
              title text,
              embedding vector(1024)
            );
            """)
        # table which contains the sections of all videos including their screen text, transcript, start time
        self.db_connection.execute(
            """
            CREATE TABLE IF NOT EXISTS video_segments (
              id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
              media_record_id uuid,
              text text,
              transcript text,
              start_time int,
              thumbnail bytea,
              title text,
              embedding vector(1024)
            );
            """)
        # table which contains links between segments of assessment textual representations
        self.db_connection.execute(
            """
            CREATE TABLE IF NOT EXISTS assessment_segments (
                task_id uuid PRIMARY KEY,
                assessment_id uuid,
                text text,
                embedding vector(1024)
            );
            """
        )
        # table which contains links between segments of different media records
        # we can't use foreign keys here because the segments live in multiple tables
        self.db_connection.execute(
            """
            CREATE TABLE IF NOT EXISTS media_record_links (
              content_id uuid,
              segment1_id uuid,
              segment2_id uuid
            );
            """)

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
                    text,
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

    def upsert_assessment_segment(self,
                                  task_id: UUID,
                                  assessment_id: UUID,
                                  textual_representation: str,
                                  embedding: Tensor) -> None:
        # Use an upsert here instead of a regular insert because the primary key of the table isn't auto-generated but
        # instead manually set. Not using an upsert might result in exceptions in case we process something twice.
        self.db_connection.execute(
            query=
            """
            INSERT INTO assessment_segments (
                task_id,
                assessment_id,
                text,
                embedding
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (task_id) 
            DO UPDATE SET 
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding;
            """,
            params=(task_id, assessment_id, textual_representation, embedding)
        )

    def delete_assessment_segments_by_assessment_id(self, assessment_id: UUID) -> list[AssessmentSegmentEntity]:
        query_results = self.db_connection.execute(
            query=
            """
            DELETE FROM assessment_segments
            WHERE assessment_id = %s
            RETURNING *;
            """,
            params=(assessment_id,)
        ).fetchall()

        return [SegmentDbConnector.__assessment_segment_query_result_to_object(result) for result in query_results]

    def insert_media_record_segment_link(self, content_id: UUID, segment1_id: UUID, segment2_id: UUID) -> None:
        self.db_connection.execute(
            query="""
                  INSERT INTO media_record_links (content_id, segment1_id, segment2_id)
                  VALUES (%s, %s, %s)
                  """,
            params=(content_id, segment1_id, segment2_id))

    def delete_media_record_segment_links_by_segment_ids(self, segment_ids: list[UUID]) -> list[MediaRecordSegmentLinkEntity]:
        query_result = self.db_connection.execute(
            """
                  DELETE FROM media_record_links 
                  WHERE segment1_id = ANY(%(segment_ids)s) OR segment2_id = ANY(%(segment_ids)s)
                  RETURNING *
                  """,
            {"segment_ids": segment_ids}).fetchall()

        return [SegmentDbConnector.__media_record_segment_link_query_result_to_object(x) for x in query_result]

    def delete_media_record_segment_links_by_content_ids(self, content_ids: list[UUID]) \
            -> list[MediaRecordSegmentLinkEntity]:
        query_result = self.db_connection.execute(
            """
            DELETE FROM media_record_links
            WHERE content_id = ANY(%(content_ids)s)
            RETURNING *;
            """, {"content_ids": content_ids}).fetchall()
        return [SegmentDbConnector.__media_record_segment_link_query_result_to_object(x) for x in query_result]

    def delete_document_segments_by_media_record_id(self, media_record_ids: list[UUID]) -> list[DocumentSegmentEntity]:
        query_results = self.db_connection.execute(
            """
                  DELETE FROM document_segments
                  WHERE media_record_id = ANY(%(mediaRecordIds)s)
                  RETURNING *
                  """,
            {"mediaRecordIds": media_record_ids})

        return [SegmentDbConnector.__document_segment_query_result_to_object(x) for x in query_results]

    def delete_video_segments_by_media_record_id(self, media_record_ids: list[UUID]) -> list[VideoSegmentEntity]:
        query_results = self.db_connection.execute(
            """
                  DELETE FROM video_segments
                  WHERE media_record_id = ANY(%(mediaRecordIds)s)
                  RETURNING *
                  """,
            {"mediaRecordIds": media_record_ids})

        return [SegmentDbConnector.__video_segment_query_result_to_object(x) for x in query_results]

    def get_segment_links_by_content_id(self, content_id: UUID) -> list[MediaRecordSegmentLinkEntity]:
        result = self.db_connection.execute("""
                    SELECT
                        segment1_id,
                        segment2_id,
                        content_id
                    FROM media_record_links WHERE content_id = %s
                    """, (content_id,)).fetchall()

        return [SegmentDbConnector.__media_record_segment_link_query_result_to_object(x) for x in result]

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

    def get_top_segments_by_embedding_distance(self, query_embedding: Tensor,
                                               count: int,
                                               parent_id_whitelist: list[UUID]) \
            -> list[SemanticSearchResultEntity]:
        # sql query to get the closest embeddings to the query embedding, both from the video and document tables
        query = """
                    WITH document_results AS (
                        SELECT
                            *,
                            'document' AS source,
                            embedding <=> %(query_embedding)s AS score
                        FROM document_segments
                        WHERE media_record_id = ANY(%(parentIdWhitelist)s)
                    ),
                    video_results AS (
                        SELECT 
                            *,
                            'video' AS source,
                            embedding <=> %(query_embedding)s AS score
                        FROM video_segments
                        WHERE media_record_id = ANY(%(parentIdWhitelist)s)
                    ),
                    assessment_results AS (
                        SELECT assessment_id, MIN(score) as score, 'assessment' As source
                        FROM (
                            SELECT
                                *,
                                embedding <=> %(query_embedding)s AS score
                            FROM assessment_segments
                            WHERE assessment_id = ANY(%(parentIdWhitelist)s)
                        )
                        GROUP BY assessment_id
                    )
                    SELECT * 
                    FROM document_results NATURAL FULL JOIN video_results NATURAL FULL JOIN assessment_results 
                    ORDER BY score LIMIT %(count)s
                """

        query_results = self.db_connection.execute(query, {
            "query_embedding": query_embedding,
            "count": count,
            "parentIdWhitelist": [str(x) for x in parent_id_whitelist],
        }).fetchall()

        return [SegmentDbConnector.__entity_semantic_search_query_result_to_object(x) for x in query_results]

    def get_media_record_segments_by_media_record_ids(self, media_record_ids: list[UUID]) \
            -> list[DocumentSegmentEntity | VideoSegmentEntity]:
        query = """
                WITH document_results AS (
                    SELECT
                        *,
                        'document' AS source
                    FROM document_segments
                    WHERE media_record_id = ANY(%(mediaRecordIds)s)
                ),
                video_results AS (
                    SELECT 
                        *,
                        'video' AS source
                    FROM video_segments
                    WHERE media_record_id = ANY(%(mediaRecordIds)s)
                )
                SELECT * FROM document_results NATURAL FULL JOIN video_results;
                """
        return self.__get_record_segments_with_query(query, {"mediaRecordIds": media_record_ids})

    def get_all_media_record_segments(self) -> list[EntitySegmentEntity]:
        query = """
                SELECT * FROM (
                    (SELECT *, 'document' AS source FROM document_segments) AS t1
                    NATURAL FULL JOIN
                    (SELECT *, 'video' AS source FROM video_segments) AS t2
                );
                """
        return self.__get_record_segments_with_query(query, {})

    def get_all_entity_segments(self) -> list[EntitySegmentEntity]:
        query = """
                SELECT * FROM (
                    (SELECT *, 'document' AS source FROM document_segments) AS t1
                    NATURAL FULL JOIN
                    (SELECT *, 'video' AS source FROM video_segments) AS t2
                    NATURAL FULL JOIN 
                    (SELECT *, 'assessment' AS source FROM assessment_segments) AS t3
                );
                """
        return self.__get_record_segments_with_query(query, {})


    def get_entity_segments_by_ids(self, segment_ids: list[UUID]) -> list[EntitySegmentEntity]:
        query = """
                WITH document_results AS (
                    SELECT
                        *,
                        'document' AS source
                    FROM document_segments
                    WHERE id = ANY(%(segmentIds)s)
                ),
                video_results AS (
                    SELECT 
                        *,
                        'video' AS source
                    FROM video_segments
                    WHERE id = ANY(%(segmentIds)s)
                ),
                assessment_results AS (
                    SELECT
                        *,
                        'assessment' AS source
                    FROM assessment_segments
                    WHERE task_id = ANY(%(segmentIds)s)
                )
                SELECT * FROM document_results NATURAL FULL JOIN video_results NATURAL FULL JOIN assessment_results;
                """
        return self.__get_record_segments_with_query(query, {"segmentIds": segment_ids})

    def __get_record_segments_with_query(self, query: str, params: dict) \
            -> list[EntitySegmentEntity]:
        query_results = self.db_connection.execute(query=query, params=params).fetchall()

        entities = []
        for x in query_results:
            entity = SegmentDbConnector.__entity_segment_query_result_to_object(x)
            entities.append(entity)

        return entities

    @staticmethod
    def __entity_semantic_search_query_result_to_object(query_result) -> SemanticSearchResultEntity:
        if query_result["source"] == "assessment":
            return SegmentDbConnector.__assessment_semantic_search_query_result_to_object(query_result)
        else:
            return MediaRecordSegmentSemanticSearchResultEntity(
                query_result["score"],
                SegmentDbConnector.__media_record_segment_query_result_to_object(query_result)
            )

    @staticmethod
    def __entity_segment_query_result_to_object(query_result) -> EntitySegmentEntity:
        if query_result["source"] == "assessment":
            return SegmentDbConnector.__assessment_segment_query_result_to_object(query_result)
        else:
            return SegmentDbConnector.__media_record_segment_query_result_to_object(query_result)

    @staticmethod
    def __media_record_segment_query_result_to_object(query_result) -> DocumentSegmentEntity | VideoSegmentEntity:
        if query_result["source"] == "document":
            return SegmentDbConnector.__document_segment_query_result_to_object(query_result)
        elif query_result["source"] == "video":
            return SegmentDbConnector.__video_segment_query_result_to_object(query_result)

    @staticmethod
    def __document_segment_query_result_to_object(query_result) -> DocumentSegmentEntity:
        return DocumentSegmentEntity(query_result["id"], query_result["media_record_id"], query_result["page"],
                                     query_result["text"], bytes(query_result["thumbnail"]), query_result["title"],
                                     query_result["embedding"])

    @staticmethod
    def __video_segment_query_result_to_object(query_result) -> VideoSegmentEntity:
        return VideoSegmentEntity(query_result["id"], query_result["media_record_id"], query_result["start_time"],
                                  query_result["transcript"], query_result["text"],
                                  bytes(query_result["thumbnail"]), query_result["title"], query_result["embedding"])

    @staticmethod
    def __assessment_segment_query_result_to_object(query_result) -> AssessmentSegmentEntity:
        return AssessmentSegmentEntity(query_result["task_id"], query_result["assessment_id"],
                                       query_result["text"], query_result["embedding"])

    @staticmethod
    def __assessment_semantic_search_query_result_to_object(query_result) -> AssessmentSemanticSearchResultEntity:
        return AssessmentSemanticSearchResultEntity(query_result["score"], query_result["assessment_id"])

    @staticmethod
    def __media_record_segment_link_query_result_to_object(query_result) -> MediaRecordSegmentLinkEntity:
        return MediaRecordSegmentLinkEntity(query_result["content_id"], query_result["segment1_id"],
                                            query_result["segment2_id"])

    def __del__(self):
        if self.db_connection is not None:
            self.db_connection.close()
