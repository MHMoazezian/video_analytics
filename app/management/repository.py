"""PostgreSQL persistence for compact facts and precomputed location rollups."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from typing import Iterator, Sequence

from app.management.models import CameraMinute

try:
    from psycopg import Connection
    from psycopg import OperationalError
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool, PoolTimeout
except ImportError:  # pragma: no cover - exercised only without the API extra
    Connection = object  # type: ignore[assignment,misc]
    ConnectionPool = None  # type: ignore[assignment]
    OperationalError = RuntimeError  # type: ignore[assignment,misc]
    PoolTimeout = RuntimeError  # type: ignore[assignment,misc]


class AnalyticsUnavailable(RuntimeError):
    """Raised when the analytical PostgreSQL store is not configured or reachable."""


class AnalyticsRepository:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.environ.get("ANALYTICS_DATABASE_URL") or os.environ.get("DATABASE_URL")
        self.pool = None
        if self.database_url and ConnectionPool is not None:
            self.pool = ConnectionPool(
                conninfo=self.database_url,
                min_size=int(os.environ.get("ANALYTICS_DB_POOL_MIN", "2")),
                max_size=int(os.environ.get("ANALYTICS_DB_POOL_MAX", "12")),
                timeout=float(os.environ.get("ANALYTICS_DB_POOL_TIMEOUT", "5")),
                open=False,
                kwargs={"row_factory": dict_row, "autocommit": False},
            )

    def open(self) -> None:
        if self.pool is not None:
            self.pool.open(wait=False)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        if self.pool is None:
            raise AnalyticsUnavailable("ANALYTICS_DATABASE_URL and psycopg pool are required")
        try:
            with self.pool.connection() as connection:
                yield connection
        except (OperationalError, PoolTimeout) as exc:
            raise AnalyticsUnavailable("analytical PostgreSQL store is unavailable") from exc

    def health(self) -> bool:
        """Report store availability without ever raising.

        ``GET /health`` must answer 200 ``degraded`` for every database problem,
        including a schema that has not been migrated yet (``UndefinedTable``).
        """

        try:
            with self.connection() as connection, connection.cursor() as cursor:
                cursor.execute("SELECT 1 FROM analytics_camera_source LIMIT 1")
            return True
        except Exception:  # noqa: BLE001 - a health probe reports, it never fails
            return False

    def ingest(
        self,
        observations: Sequence[CameraMinute],
        sla_target_seconds: float,
        *,
        sync_sources: bool = True,
    ) -> int:
        with self.connection() as connection, connection.cursor() as cursor:
            if sync_sources:
                cursor.execute("SELECT analytics_sync_camera_sources()")
            for item in observations:
                cursor.execute(
                    """
                    INSERT INTO analytics_camera_source (camera_id, camera_name, physically_calibrated)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (camera_id) DO UPDATE SET
                      camera_name = COALESCE(EXCLUDED.camera_name, analytics_camera_source.camera_name),
                      physically_calibrated = analytics_camera_source.physically_calibrated OR EXCLUDED.physically_calibrated,
                      updated_at = now()
                    """,
                    (item.camera_id, item.camera_name or item.camera_id, item.physically_calibrated),
                )
                cursor.execute(
                    """
                    INSERT INTO analytics_camera_minute
                      (camera_id, bucket_start, sample_count, expected_samples, confidence_sum,
                       occupancy_sum, occupancy_max, occupancy_last, entries, exits, unique_visitors)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (camera_id, bucket_start) DO UPDATE SET
                      sample_count=EXCLUDED.sample_count, expected_samples=EXCLUDED.expected_samples,
                      confidence_sum=EXCLUDED.confidence_sum, occupancy_sum=EXCLUDED.occupancy_sum,
                      occupancy_max=EXCLUDED.occupancy_max, occupancy_last=EXCLUDED.occupancy_last,
                      entries=EXCLUDED.entries, exits=EXCLUDED.exits,
                      unique_visitors=EXCLUDED.unique_visitors, updated_at=now()
                    """,
                    (item.camera_id, item.bucket_start, item.sample_count, item.expected_samples,
                     item.confidence_sum, item.occupancy_sum, item.occupancy_max,
                     item.occupancy_last, item.entries, item.exits, item.unique_visitors),
                )
                for queue in item.queues:
                    cursor.execute(
                        """
                        INSERT INTO analytics_queue_minute
                          (camera_id,queue_id,queue_name,bucket_start,sample_count,length_sum,length_max,
                           length_last,wait_sum_seconds,wait_sample_count,throughput,sla_met,warning_samples,
                           critical_samples,movement_speed_sum_mpm,movement_speed_samples,physically_calibrated)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (camera_id,queue_id,bucket_start) DO UPDATE SET
                          queue_name=EXCLUDED.queue_name,sample_count=EXCLUDED.sample_count,
                          length_sum=EXCLUDED.length_sum,length_max=EXCLUDED.length_max,length_last=EXCLUDED.length_last,
                          wait_sum_seconds=EXCLUDED.wait_sum_seconds,wait_sample_count=EXCLUDED.wait_sample_count,
                          throughput=EXCLUDED.throughput,sla_met=EXCLUDED.sla_met,
                          warning_samples=EXCLUDED.warning_samples,critical_samples=EXCLUDED.critical_samples,
                          movement_speed_sum_mpm=EXCLUDED.movement_speed_sum_mpm,
                          movement_speed_samples=EXCLUDED.movement_speed_samples,
                          physically_calibrated=EXCLUDED.physically_calibrated,updated_at=now()
                        """,
                        (item.camera_id, queue.queue_id, queue.queue_name, item.bucket_start,
                         queue.sample_count, queue.length_sum, queue.length_max, queue.length_last,
                         queue.wait_sum_seconds, queue.wait_sample_count, queue.throughput, queue.sla_met,
                         queue.warning_samples, queue.critical_samples, queue.movement_speed_sum_mpm,
                         queue.movement_speed_samples, queue.physically_calibrated),
                    )
                    for index, wait in enumerate(queue.completed_wait_seconds):
                        identity = f"{item.camera_id}:{queue.queue_id}:{item.bucket_start.isoformat()}:{index}:{wait}"
                        event_id = hashlib.sha256(identity.encode()).hexdigest()
                        cursor.execute(
                            """INSERT INTO analytics_queue_wait_event
                               (event_id,camera_id,queue_id,occurred_at,wait_seconds,sla_met)
                               VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                            (event_id, item.camera_id, queue.queue_id, item.bucket_start, wait, wait <= sla_target_seconds),
                        )
                for spatial in item.spatial:
                    cursor.execute(
                        """
                        INSERT INTO analytics_spatial_bucket
                          (camera_id,zone_id,zone_name,layer,bucket_start,bucket_seconds,points,coverage_percent)
                        VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s)
                        ON CONFLICT (camera_id,zone_id,layer,bucket_start) DO UPDATE SET
                          zone_name=EXCLUDED.zone_name,bucket_seconds=EXCLUDED.bucket_seconds,
                          points=EXCLUDED.points,coverage_percent=EXCLUDED.coverage_percent,updated_at=now()
                        """,
                        (item.camera_id, spatial.zone_id, spatial.zone_name, spatial.layer,
                         item.bucket_start, spatial.bucket_seconds,
                         json.dumps([point.model_dump(by_alias=True) for point in spatial.points]),
                         spatial.coverage_percent),
                    )
                for event in item.events:
                    cursor.execute(
                        """
                        INSERT INTO analytics_event
                          (event_id,camera_id,event_type,severity,title,metric_key,occurred_at,value,payload)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                        ON CONFLICT DO NOTHING
                        """,
                        (event.event_id, item.camera_id, event.event_type, event.severity,
                         event.title, event.metric_key, event.occurred_at, event.value,
                         json.dumps(event.payload) if event.payload is not None else None),
                    )
            connection.commit()
        return len(observations)

    def refresh_rollups(self, lookback_hours: int = 3) -> bool:
        now = datetime.now(timezone.utc)
        with self.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT analytics_refresh_location_rollups(%s, %s) AS refreshed",
                (now - timedelta(hours=lookback_hours), now + timedelta(hours=1)),
            )
            row = cursor.fetchone()
            cursor.execute("SELECT analytics_purge_expired()")
            connection.commit()
            return bool(row and row["refreshed"])

    def rows(self, sql: str, params: Sequence[object] = ()) -> list[dict[str, object]]:
        with self.connection() as connection, connection.cursor() as cursor:
            cursor.execute(sql, params)
            return list(cursor.fetchall())

    def row(self, sql: str, params: Sequence[object] = ()) -> dict[str, object] | None:
        values = self.rows(sql, params)
        return values[0] if values else None


analytics_repository = AnalyticsRepository()


def location_predicate(alias: str, location_type: str, location_id: str | None) -> tuple[str, list[object]]:
    if location_type == "organization":
        return "TRUE", []
    if not location_id:
        raise ValueError("locationId is required")
    column = {"field": "field_id", "market": "market_id", "booth": "booth_id"}[location_type]
    return f"{alias}.{column} = %s", [location_id]


def rollup_location_predicate(alias: str, target_type: str, location_type: str, location_id: str | None) -> tuple[str, list[object]]:
    if location_type == "organization":
        return "TRUE", []
    if not location_id:
        raise ValueError("locationId is required")
    if target_type == location_type:
        return f"{alias}.location_id = %s", [location_id]
    target_column = {"field": "field_id", "market": "market_id", "booth": "booth_id"}[target_type]
    selected_column = {"field": "field_id", "market": "market_id", "booth": "booth_id"}[location_type]
    return (
        f"EXISTS (SELECT 1 FROM analytics_camera_source scope "
        f"WHERE scope.{target_column} = {alias}.location_id AND scope.{selected_column} = %s)",
        [location_id],
    )
