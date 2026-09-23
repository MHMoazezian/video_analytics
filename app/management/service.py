"""Location-centric analytical queries over bounded PostgreSQL rollups."""

from __future__ import annotations

from collections import defaultdict
import calendar
from datetime import datetime, timedelta, timezone, tzinfo
import os
from typing import Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.management.models import AnalyticsQuery
from app.management.repository import (
    AnalyticsRepository,
    analytics_repository,
    location_predicate,
    rollup_location_predicate,
)


def _number(value: object) -> float | None:
    return float(value) if value is not None else None


def _metric(value: float | int | None, previous: float | int | None = None) -> dict[str, float | None]:
    change = None
    if value is not None and previous not in (None, 0):
        change = round((float(value) - float(previous)) * 100 / float(previous), 2)
    return {"value": None if value is None else round(float(value), 2), "changePercent": change}


def _period(query: AnalyticsQuery) -> tuple[datetime, datetime]:
    start = query.from_date.astimezone(timezone.utc)
    end = query.to_date.astimezone(timezone.utc)
    return start, end


def _comparison(query: AnalyticsQuery) -> tuple[datetime, datetime] | None:
    start, end = _period(query)
    if query.comparison == "none":
        return None
    if query.comparison == "previous_week":
        return start - timedelta(days=7), end - timedelta(days=7)
    if query.comparison == "previous_month":
        return _shift_month(start), _shift_month(end)
    duration = end - start
    return start - duration, start


def _shift_month(value: datetime) -> datetime:
    year = value.year if value.month > 1 else value.year - 1
    month = value.month - 1 if value.month > 1 else 12
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _business_zone() -> tzinfo:
    """Zone used for human-readable labels; mirrors the query-date handling."""

    try:
        return ZoneInfo(os.environ.get("ANALYTICS_TIMEZONE", "Asia/Tehran"))
    except (ValueError, ZoneInfoNotFoundError):
        return timezone.utc


def _target_type(query: AnalyticsQuery) -> str:
    if query.place_type != "all":
        return query.place_type
    return {"organization": "field", "field": "market", "market": "booth", "booth": "booth"}[query.location_type]


def _overall_rollup_type(query: AnalyticsQuery) -> str:
    return "field" if query.location_type == "organization" else query.location_type


def _time_predicate(alias: str, column: str, query: AnalyticsQuery) -> tuple[str, list[object]]:
    if not query.time_from and not query.time_to:
        return "TRUE", []
    timezone_name = os.environ.get("ANALYTICS_TIMEZONE", "Asia/Tehran")
    expression = f"({alias}.{column} AT TIME ZONE %s)::time"
    if query.time_from and query.time_to:
        if query.time_from <= query.time_to:
            return f"{expression} >= %s::time AND {expression} <= %s::time", [timezone_name, query.time_from, timezone_name, query.time_to]
        return f"({expression} >= %s::time OR {expression} <= %s::time)", [timezone_name, query.time_from, timezone_name, query.time_to]
    if query.time_from:
        return f"{expression} >= %s::time", [timezone_name, query.time_from]
    return f"{expression} <= %s::time", [timezone_name, query.time_to or "23:59"]


def _quality(row: dict[str, object] | None, expected_sources: int, calibrated: int | None = None) -> dict[str, object]:
    row = row or {}
    samples = float(row.get("sample_count") or 0)
    expected = float(row.get("expected_samples") or 0)
    confidence_sum = float(row.get("confidence_sum") or 0)
    coverage = min(100.0, samples * 100 / expected) if expected > 0 else None
    confidence = confidence_sum * 100 / samples if samples > 0 else None
    result: dict[str, object] = {
        "coveragePercent": round(coverage, 2) if coverage is not None else None,
        "confidencePercent": round(confidence, 2) if confidence is not None else None,
        "contributingSources": int(row.get("contributing_sources") or 0),
        "expectedSources": expected_sources,
    }
    if calibrated is not None:
        result["calibratedSources"] = calibrated
    if expected_sources and result["contributingSources"] < expected_sources:
        result["note"] = "Some configured sources did not contribute; metrics are computed from available rollups."
    return result


def _status(quality: dict[str, object]) -> str:
    coverage = quality["coveragePercent"]
    if coverage is None or float(coverage) <= 0:
        return "unavailable"
    return "live" if float(coverage) >= 95 else "partial"


class ManagementAnalyticsService:
    def __init__(self, repository: AnalyticsRepository = analytics_repository) -> None:
        self.repository = repository

    def people_flow(self, query: AnalyticsQuery) -> dict[str, object]:
        start, end = _period(query)
        previous_period = _comparison(query)
        overall_type = _overall_rollup_type(query)
        current = self._flow_summary(query, start, end, overall_type)
        previous = self._flow_summary(query, *previous_period, overall_type) if previous_period else {}
        expected_sources = self._expected_sources(query)
        quality = _quality(current, expected_sources)
        latest = self._latest_occupancy(query)
        trend_current = self._flow_trend(query, start, end, overall_type)
        trend_previous = self._flow_trend(query, *previous_period, overall_type) if previous_period else []
        previous_by_index = [row.get("occupancy") for row in trend_previous]
        trend = [{
            "timestamp": row["timestamp"].isoformat(),
            "occupancy": _number(row.get("occupancy")),
            "entries": int(row.get("entries") or 0),
            "exits": int(row.get("exits") or 0),
            "uniqueVisitors": row.get("unique_visitors"),
            "previousOccupancy": _number(previous_by_index[index]) if index < len(previous_by_index) else None,
        } for index, row in enumerate(trend_current)]
        locations = self._flow_locations(query, start, end, _target_type(query))
        events = self._events(query, start, end)
        return {
            "generatedAt": datetime.now(timezone.utc).isoformat(), "dataStatus": _status(quality),
            "dataQuality": quality, "bucket": query.bucket,
            "uniqueVisitorsAggregation": "location_period" if expected_sources == 1 else "not_available",
            "kpis": {
                "currentOccupancy": _metric(latest, None),
                "averageOccupancy": _metric(_number(current.get("average_occupancy")), _number(previous.get("average_occupancy"))),
                "peakOccupancy": _metric(_number(current.get("peak_occupancy")), _number(previous.get("peak_occupancy"))),
                "entries": _metric(_number(current.get("entries")), _number(previous.get("entries"))),
                "exits": _metric(_number(current.get("exits")), _number(previous.get("exits"))),
                "uniqueVisitors": _metric(_number(current.get("unique_visitors")) if expected_sources == 1 else None, _number(previous.get("unique_visitors")) if expected_sources == 1 else None),
            },
            "trend": trend,
            "peakPeriods": self._peak_periods(query, start, end, overall_type),
            "activityHeatmap": self._activity_heatmap(query, start, end, overall_type),
            "locations": locations, "events": events,
        }

    def queues(self, query: AnalyticsQuery) -> dict[str, object]:
        start, end = _period(query)
        previous_period = _comparison(query)
        overall_type = _overall_rollup_type(query)
        current = self._queue_summary(query, start, end, overall_type)
        previous = self._queue_summary(query, *previous_period, overall_type) if previous_period else {}
        expected_sources = self._expected_sources(query)
        quality_row = self._flow_summary(query, start, end, overall_type)
        calibrated = int(current.get("calibrated_sources") or 0)
        quality = _quality(quality_row, expected_sources, calibrated)
        percentiles = self._wait_percentiles(query, start, end)
        previous_percentiles = self._wait_percentiles(query, *previous_period) if previous_period else {}
        latest = self._latest_queue_length(query)
        sample_seconds = float(os.environ.get("ANALYTICS_SAMPLE_INTERVAL_SECONDS", "2"))
        trend_current = self._queue_trend(query, start, end, overall_type)
        trend_previous = self._queue_trend(query, *previous_period, overall_type) if previous_period else []
        trend = [{
            "timestamp": row["timestamp"].isoformat(),
            "averageLength": _number(row.get("average_length")),
            "averageWaitMinutes": _seconds_to_minutes(row.get("average_wait_seconds")),
            "throughput": int(row.get("throughput") or 0),
            "slaPercent": _ratio(row.get("sla_met"), row.get("throughput")),
            "previousWaitMinutes": _seconds_to_minutes(trend_previous[index].get("average_wait_seconds")) if index < len(trend_previous) else None,
        } for index, row in enumerate(trend_current)]
        movement = _ratio(current.get("movement_speed_sum_mpm"), current.get("movement_speed_samples"), scale=1)
        kpis = {
            "currentLength": _metric(latest),
            "averageLength": _metric(_number(current.get("average_length")), _number(previous.get("average_length"))),
            "maximumLength": _metric(_number(current.get("maximum_length")), _number(previous.get("maximum_length"))),
            "averageWaitMinutes": _metric(_seconds_to_minutes(current.get("average_wait_seconds")), _seconds_to_minutes(previous.get("average_wait_seconds"))),
            "p50WaitMinutes": _metric(_seconds_to_minutes(percentiles.get("p50")), _seconds_to_minutes(previous_percentiles.get("p50"))),
            "p90WaitMinutes": _metric(_seconds_to_minutes(percentiles.get("p90")), _seconds_to_minutes(previous_percentiles.get("p90"))),
            "p95WaitMinutes": _metric(_seconds_to_minutes(percentiles.get("p95")), _seconds_to_minutes(previous_percentiles.get("p95"))),
            "movementSpeedMetersPerMinute": _metric(movement if calibrated else None),
            "throughputPerHour": _metric(_per_hour(current.get("throughput"), start, end), _per_hour(previous.get("throughput"), *(previous_period or (start, end)))),
            "slaPercent": _metric(_ratio(current.get("sla_met"), current.get("throughput")), _ratio(previous.get("sla_met"), previous.get("throughput"))),
            "warningMinutes": _metric(float(current.get("warning_samples") or 0) * sample_seconds / 60),
            "criticalMinutes": _metric(float(current.get("critical_samples") or 0) * sample_seconds / 60),
        }
        return {
            "generatedAt": datetime.now(timezone.utc).isoformat(), "dataStatus": _status(quality),
            "dataQuality": quality, "bucket": query.bucket,
            "slaTargetMinutes": float(os.environ.get("ANALYTICS_QUEUE_SLA_SECONDS", "300")) / 60,
            "kpis": kpis, "trend": trend,
            "locations": self._queue_locations(query, start, end, _target_type(query), sample_seconds),
            "queues": self._queue_details(query, start, end, sample_seconds),
            "events": self._events(query, start, end),
        }

    def spatial(self, query: AnalyticsQuery) -> dict[str, object]:
        start, end = _period(query)
        comparison = _comparison(query)
        current_rows = self._spatial_rows(query, start, end)
        previous_rows = self._spatial_rows(query, *comparison) if comparison else []
        layers = []
        for layer in ("occupancy", "dwell", "traffic", "congestion"):
            current_points = _merge_spatial(current_rows, layer)
            previous_points = _merge_spatial(previous_rows, layer)
            layers.append({
                "layer": layer,
                "unit": {"occupancy": "people", "dwell": "minutes", "traffic": "movements", "congestion": "percent"}[layer],
                "periodA": current_points, "periodB": previous_points,
                "difference": _difference_points(current_points, previous_points),
                "timeline": _spatial_timeline(current_rows, layer),
            })
        expected_sources = self._expected_sources(query)
        contributing = len({str(row["camera_id"]) for row in current_rows})
        coverages = [float(row["coverage_percent"]) for row in current_rows if row.get("coverage_percent") is not None]
        quality = {
            "coveragePercent": round(sum(coverages) / len(coverages), 2) if coverages else None,
            "confidencePercent": round(sum(coverages) / len(coverages), 2) if coverages else None,
            "contributingSources": contributing, "expectedSources": expected_sources,
        }
        if contributing < expected_sources:
            quality["note"] = "Spatial output includes only sources mapped to the shared location coordinate system."
        return {
            "generatedAt": datetime.now(timezone.utc).isoformat(), "dataStatus": _status(quality),
            "dataQuality": quality, "coordinateSystem": "location_normalized_percent",
            "periodA": {"from": start.date().isoformat(), "to": (end - timedelta(microseconds=1)).date().isoformat()},
            "periodB": {"from": comparison[0].date().isoformat(), "to": (comparison[1] - timedelta(microseconds=1)).date().isoformat()} if comparison else None,
            "layers": layers, "zones": _zone_metrics(current_rows), "events": self._events(query, start, end),
        }

    def overview(self, query: AnalyticsQuery) -> dict[str, object]:
        flow = self.people_flow(query)
        queue = self.queues(query)
        spatial = self.spatial(query)
        return {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "dataStatus": flow["dataStatus"],
            "kpis": {
                "visitors": flow["kpis"]["uniqueVisitors"],
                "entries": flow["kpis"]["entries"],
                "peakOccupancy": flow["kpis"]["peakOccupancy"],
                "averageDwellMinutes": _metric(_average_dwell(spatial["zones"])),
                "averageWaitMinutes": queue["kpis"]["averageWaitMinutes"],
                "queueSlaPercent": queue["kpis"]["slaPercent"],
            },
            "trend": [{"timestamp": row["timestamp"], "traffic": row["entries"], "occupancy": row["occupancy"]} for row in flow["trend"]],
            "alerts": _alerts(flow["events"]),
            "activityHeatmap": flow["activityHeatmap"],
            "queue": _queue_summary_card(queue["kpis"], queue["queues"]),
            "ranking": _ranking(flow["locations"], queue["locations"]),
            "spatialHeatmap": [{"x": point["x"], "y": point["y"], "intensity": point["intensity"]}
                                for point in _merge_spatial(self._spatial_rows(query, *_period(query)), "occupancy")],
            "importantChanges": _important_changes(flow["kpis"], queue["kpis"]),
        }

    def _expected_sources(self, query: AnalyticsQuery) -> int:
        predicate, params = location_predicate("s", query.location_type, query.location_id)
        row = self.repository.row(f"SELECT count(*) AS count FROM analytics_camera_source s WHERE s.enabled AND {predicate}", params)
        return int(row["count"] if row else 0)

    def _flow_summary(self, query: AnalyticsQuery, start: datetime, end: datetime, rollup_type: str) -> dict[str, object]:
        predicate, params = rollup_location_predicate("h", rollup_type, query.location_type, query.location_id)
        time_sql, time_params = _time_predicate("h", "bucket_start", query)
        return self.repository.row(f"""
          WITH selected AS (
            SELECT * FROM analytics_location_hour h WHERE h.location_type=%s AND h.bucket_start >= %s AND h.bucket_start < %s AND {predicate} AND {time_sql}
          ), bucket_totals AS (
            SELECT bucket_start,SUM(occupancy_sum) occupancy,SUM(occupancy_max) peak,
                   SUM(contributing_sources) contributing FROM selected GROUP BY bucket_start
          )
          SELECT SUM(sample_count) sample_count,SUM(expected_samples) expected_samples,
            SUM(confidence_sum) confidence_sum,(SELECT MAX(contributing) FROM bucket_totals) contributing_sources,
            (SELECT AVG(occupancy) FROM bucket_totals) average_occupancy,
            (SELECT MAX(peak) FROM bucket_totals) peak_occupancy,SUM(entries) entries,SUM(exits) exits,
            CASE WHEN MAX(expected_sources)=1 THEN MAX(unique_visitors) END unique_visitors
          FROM selected
        """, [rollup_type, start, end, *params, *time_params]) or {}

    def _flow_trend(self, query: AnalyticsQuery, start: datetime, end: datetime, rollup_type: str) -> list[dict[str, object]]:
        predicate, params = rollup_location_predicate("h", rollup_type, query.location_type, query.location_id)
        time_sql, time_params = _time_predicate("h", "bucket_start", query)
        trunc = "day" if query.bucket == "day" else "hour"
        occupancy_aggregation = "AVG(occupancy)" if query.bucket == "day" else "SUM(occupancy)"
        return self.repository.rows(f"""
          WITH hourly AS (
            SELECT bucket_start,SUM(occupancy_sum) occupancy,SUM(entries) entries,SUM(exits) exits,
              CASE WHEN MAX(expected_sources)=1 THEN MAX(unique_visitors) END unique_visitors
            FROM analytics_location_hour h
            WHERE h.location_type=%s AND bucket_start >= %s AND bucket_start < %s AND {predicate} AND {time_sql}
            GROUP BY bucket_start
          )
          SELECT date_trunc('{trunc}',bucket_start) timestamp,{occupancy_aggregation} occupancy,
            SUM(entries) entries,SUM(exits) exits,MAX(unique_visitors) unique_visitors
          FROM hourly
          GROUP BY 1 ORDER BY 1
        """, [rollup_type, start, end, *params, *time_params])

    def _latest_occupancy(self, query: AnalyticsQuery) -> int | None:
        predicate, params = location_predicate("s", query.location_type, query.location_id)
        row = self.repository.row(f"""
          WITH latest AS (SELECT DISTINCT ON (m.camera_id) m.camera_id,m.occupancy_last
            FROM analytics_camera_minute m JOIN analytics_camera_source s USING(camera_id)
            WHERE m.bucket_start >= now()-interval '10 minutes' AND s.enabled AND {predicate}
            ORDER BY m.camera_id,m.bucket_start DESC)
          SELECT SUM(occupancy_last) value FROM latest
        """, params)
        return int(row["value"]) if row and row["value"] is not None else None

    def _flow_locations(self, query: AnalyticsQuery, start: datetime, end: datetime, target: str) -> list[dict[str, object]]:
        predicate, params = rollup_location_predicate("h", target, query.location_type, query.location_id)
        time_sql, time_params = _time_predicate("h", "bucket_start", query)
        rows = self.repository.rows(f"""
          SELECT location_id,MAX(location_name) name,MAX(parent_name) parent_name,
            AVG(occupancy_sum) value,
            MAX(occupancy_max) current_occupancy,SUM(entries) entries,SUM(exits) exits,
            CASE WHEN MAX(expected_sources)=1 THEN MAX(unique_visitors) END unique_visitors,
            LEAST(100,SUM(sample_count)*100.0/NULLIF(SUM(expected_samples),0)) coverage_percent
          FROM analytics_location_hour h WHERE location_type=%s AND bucket_start >= %s AND bucket_start < %s AND {predicate} AND {time_sql}
          GROUP BY location_id ORDER BY value DESC NULLS LAST LIMIT 500
        """, [target, start, end, *params, *time_params])
        return [{"locationId": row["location_id"], "locationType": target, "name": row["name"],
                 "parentName": row["parent_name"], "value": _number(row["value"]), "previousValue": None,
                 "changePercent": None, "coveragePercent": _number(row["coverage_percent"]),
                 "currentOccupancy": row["current_occupancy"], "entries": row["entries"], "exits": row["exits"],
                 "uniqueVisitors": row["unique_visitors"]} for row in rows]

    def _peak_periods(self, query: AnalyticsQuery, start: datetime, end: datetime, rollup_type: str) -> list[dict[str, object]]:
        rows = self._flow_trend(query, start, end, rollup_type)
        duration = timedelta(days=1) if query.bucket == "day" else timedelta(hours=1)
        ranked = sorted((row for row in rows if row.get("occupancy") is not None), key=lambda row: float(row["occupancy"]), reverse=True)[:8]
        local_zone = _business_zone()
        return [{"from": row["timestamp"].isoformat(), "to": (row["timestamp"] + duration).isoformat(),
                 "value": round(float(row["occupancy"]), 2), "label": f"اوج {row['timestamp'].astimezone(local_zone).strftime('%H:%M')}"} for row in ranked]

    def _activity_heatmap(self, query: AnalyticsQuery, start: datetime, end: datetime, rollup_type: str) -> list[dict[str, object]]:
        predicate, params = rollup_location_predicate("h", rollup_type, query.location_type, query.location_id)
        time_sql, time_params = _time_predicate("h", "bucket_start", query)
        timezone_name = os.environ.get("ANALYTICS_TIMEZONE", "Asia/Tehran").replace("'", "")
        rows = self.repository.rows(f"""
          WITH hourly AS (
            SELECT bucket_start,SUM(occupancy_sum) value
            FROM analytics_location_hour h WHERE location_type=%s AND bucket_start >= %s AND bucket_start < %s AND {predicate} AND {time_sql}
            GROUP BY bucket_start
          )
          SELECT extract(dow from bucket_start AT TIME ZONE '{timezone_name}')::int AS day_index,
            extract(hour from bucket_start AT TIME ZONE '{timezone_name}')::int AS hour_index,
            AVG(value) value
          FROM hourly
          GROUP BY 1,2 ORDER BY 1,2
        """, [rollup_type, start, end, *params, *time_params])
        return [{"day": (int(row["day_index"]) + 1) % 7, "hour": int(row["hour_index"]), "value": _number(row["value"])} for row in rows]

    def _queue_summary(self, query: AnalyticsQuery, start: datetime, end: datetime, rollup_type: str) -> dict[str, object]:
        predicate, params = rollup_location_predicate("h", rollup_type, query.location_type, query.location_id)
        time_sql, time_params = _time_predicate("h", "bucket_start", query)
        return self.repository.row(f"""
          SELECT SUM(sample_count) sample_count,CASE WHEN SUM(sample_count)>0 THEN SUM(length_sum)/SUM(sample_count) END average_length,
            MAX(length_max) maximum_length,SUM(wait_sum_seconds)/NULLIF(SUM(wait_sample_count),0) average_wait_seconds,
            SUM(throughput) throughput,SUM(sla_met) sla_met,SUM(warning_samples) warning_samples,
            SUM(critical_samples) critical_samples,SUM(movement_speed_sum_mpm) movement_speed_sum_mpm,
            SUM(movement_speed_samples) movement_speed_samples,MAX(calibrated_sources) calibrated_sources
          FROM analytics_queue_location_hour h WHERE location_type=%s AND bucket_start >= %s AND bucket_start < %s AND {predicate} AND {time_sql}
        """, [rollup_type, start, end, *params, *time_params]) or {}

    def _queue_trend(self, query: AnalyticsQuery, start: datetime, end: datetime, rollup_type: str) -> list[dict[str, object]]:
        predicate, params = rollup_location_predicate("h", rollup_type, query.location_type, query.location_id)
        time_sql, time_params = _time_predicate("h", "bucket_start", query)
        trunc = "day" if query.bucket == "day" else "hour"
        return self.repository.rows(f"""
          SELECT date_trunc('{trunc}',bucket_start) timestamp,
            SUM(length_sum)/NULLIF(SUM(sample_count),0) average_length,
            SUM(wait_sum_seconds)/NULLIF(SUM(wait_sample_count),0) average_wait_seconds,
            SUM(throughput) throughput,SUM(sla_met) sla_met
          FROM analytics_queue_location_hour h WHERE location_type=%s AND bucket_start >= %s AND bucket_start < %s AND {predicate} AND {time_sql}
          GROUP BY 1 ORDER BY 1
        """, [rollup_type, start, end, *params, *time_params])

    def _latest_queue_length(self, query: AnalyticsQuery) -> int | None:
        predicate, params = location_predicate("s", query.location_type, query.location_id)
        row = self.repository.row(f"""
          WITH latest AS (SELECT DISTINCT ON (q.camera_id,q.queue_id) q.length_last
            FROM analytics_queue_minute q JOIN analytics_camera_source s USING(camera_id)
            WHERE q.bucket_start >= now()-interval '10 minutes' AND s.enabled AND {predicate}
            ORDER BY q.camera_id,q.queue_id,q.bucket_start DESC)
          SELECT SUM(length_last) value FROM latest
        """, params)
        return int(row["value"]) if row and row["value"] is not None else None

    def _wait_percentiles(self, query: AnalyticsQuery, start: datetime, end: datetime) -> dict[str, object]:
        predicate, params = location_predicate("s", query.location_type, query.location_id)
        time_sql, time_params = _time_predicate("w", "occurred_at", query)
        return self.repository.row(f"""
          SELECT percentile_cont(.5) WITHIN GROUP (ORDER BY w.wait_seconds) p50,
            percentile_cont(.9) WITHIN GROUP (ORDER BY w.wait_seconds) p90,
            percentile_cont(.95) WITHIN GROUP (ORDER BY w.wait_seconds) p95
          FROM analytics_queue_wait_event w JOIN analytics_camera_source s USING(camera_id)
          WHERE w.occurred_at >= %s AND w.occurred_at < %s AND {predicate} AND {time_sql}
        """, [start, end, *params, *time_params]) or {}

    def _queue_locations(self, query: AnalyticsQuery, start: datetime, end: datetime, target: str, sample_seconds: float) -> list[dict[str, object]]:
        predicate, params = rollup_location_predicate("h", target, query.location_type, query.location_id)
        time_sql, time_params = _time_predicate("h", "bucket_start", query)
        rows = self.repository.rows(f"""
          SELECT location_id,MAX(location_name) name,MAX(parent_name) parent_name,
            SUM(length_sum)/NULLIF(SUM(sample_count),0) current_length,
            SUM(wait_sum_seconds)/NULLIF(SUM(wait_sample_count),0) average_wait,
            SUM(throughput) throughput,SUM(sla_met) sla_met,SUM(warning_samples) warning_samples,
            SUM(critical_samples) critical_samples
          FROM analytics_queue_location_hour h WHERE location_type=%s AND bucket_start >= %s AND bucket_start < %s AND {predicate} AND {time_sql}
          GROUP BY location_id ORDER BY average_wait DESC NULLS LAST LIMIT 500
        """, [target, start, end, *params, *time_params])
        return [{"locationId": row["location_id"], "locationType": target, "name": row["name"],
                 "parentName": row["parent_name"], "value": _seconds_to_minutes(row["average_wait"]),
                 "previousValue": None, "changePercent": None, "coveragePercent": None,
                 "currentLength": _number(row["current_length"]), "averageWaitMinutes": _seconds_to_minutes(row["average_wait"]),
                 "p95WaitMinutes": None, "throughput": row["throughput"],
                 "slaPercent": _ratio(row["sla_met"], row["throughput"]),
                 "warningMinutes": float(row["warning_samples"] or 0)*sample_seconds/60,
                 "criticalMinutes": float(row["critical_samples"] or 0)*sample_seconds/60} for row in rows]

    def _queue_details(self, query: AnalyticsQuery, start: datetime, end: datetime, sample_seconds: float) -> list[dict[str, object]]:
        predicate, params = location_predicate("s", query.location_type, query.location_id)
        time_sql, time_params = _time_predicate("q", "bucket_start", query)
        rows = self.repository.rows(f"""
          SELECT q.queue_id,MAX(q.queue_name) name,
            COALESCE(s.booth_id,s.market_id,s.field_id) location_id,
            CASE WHEN s.booth_id IS NOT NULL THEN 'booth' WHEN s.market_id IS NOT NULL THEN 'market' ELSE 'field' END location_type,
            COALESCE(s.booth_name,s.market_name,s.field_name) location_name,
            SUM(q.length_sum)/NULLIF(SUM(q.sample_count),0) average_length,MAX(q.length_max) maximum_length,
            SUM(q.wait_sum_seconds)/NULLIF(SUM(q.wait_sample_count),0) average_wait,
            SUM(q.throughput) throughput,SUM(q.sla_met) sla_met,SUM(q.warning_samples) warning_samples,
            SUM(q.critical_samples) critical_samples,SUM(q.movement_speed_sum_mpm)/NULLIF(SUM(q.movement_speed_samples),0) speed,
            bool_or(q.physically_calibrated) calibrated
          FROM analytics_queue_minute q JOIN analytics_camera_source s USING(camera_id)
          WHERE q.bucket_start >= %s AND q.bucket_start < %s AND s.enabled AND {predicate} AND {time_sql}
          GROUP BY q.queue_id,s.booth_id,s.market_id,s.field_id,s.booth_name,s.market_name,s.field_name
          ORDER BY average_wait DESC NULLS LAST LIMIT 500
        """, [start, end, *params, *time_params])
        latest = self._latest_queue_by_id(query)
        return [{"id": f"{row['location_id']}:{row['queue_id']}", "name": row["name"],
                 "locationId": row["location_id"], "locationType": row["location_type"], "locationName": row["location_name"],
                 "currentLength": latest.get((str(row["location_id"]), str(row["queue_id"]))),
                 "averageLength": _number(row["average_length"]), "maximumLength": row["maximum_length"],
                 "averageWaitMinutes": _seconds_to_minutes(row["average_wait"]),
                 "p50WaitMinutes": None, "p90WaitMinutes": None, "p95WaitMinutes": None,
                 "movementSpeedMetersPerMinute": _number(row["speed"]) if row["calibrated"] else None,
                 "isPhysicallyCalibrated": bool(row["calibrated"]), "throughputPerHour": _per_hour(row["throughput"], start, end),
                 "slaPercent": _ratio(row["sla_met"], row["throughput"]),
                 "warningMinutes": float(row["warning_samples"] or 0)*sample_seconds/60,
                 "criticalMinutes": float(row["critical_samples"] or 0)*sample_seconds/60,
                 "coveragePercent": None} for row in rows]

    def _latest_queue_by_id(self, query: AnalyticsQuery) -> dict[tuple[str, str], int | None]:
        predicate, params = location_predicate("s", query.location_type, query.location_id)
        rows = self.repository.rows(f"""
          SELECT DISTINCT ON (q.camera_id,q.queue_id) COALESCE(s.booth_id,s.market_id,s.field_id) location_id,q.queue_id,q.length_last
          FROM analytics_queue_minute q JOIN analytics_camera_source s USING(camera_id)
          WHERE q.bucket_start >= now()-interval '10 minutes' AND {predicate}
          ORDER BY q.camera_id,q.queue_id,q.bucket_start DESC
        """, params)
        result: dict[tuple[str, str], int | None] = defaultdict(int)
        for row in rows:
            key = (str(row["location_id"]), str(row["queue_id"]))
            if row["length_last"] is not None:
                result[key] = int(result[key] or 0) + int(row["length_last"])
        return result

    def _spatial_rows(self, query: AnalyticsQuery, start: datetime, end: datetime) -> list[dict[str, object]]:
        predicate, params = location_predicate("s", query.location_type, query.location_id)
        time_sql, time_params = _time_predicate("p", "bucket_start", query)
        return self.repository.rows(f"""
          SELECT p.*,COALESCE(s.booth_id,s.market_id,s.field_id) location_id,
            CASE WHEN s.booth_id IS NOT NULL THEN 'booth' WHEN s.market_id IS NOT NULL THEN 'market' ELSE 'field' END location_type,
            COALESCE(s.booth_name,s.market_name,s.field_name) location_name
          FROM analytics_spatial_bucket p JOIN analytics_camera_source s USING(camera_id)
          WHERE p.bucket_start >= %s AND p.bucket_start < %s AND s.enabled AND {predicate} AND {time_sql}
          ORDER BY p.bucket_start
        """, [start, end, *params, *time_params])

    def _events(self, query: AnalyticsQuery, start: datetime, end: datetime) -> list[dict[str, object]]:
        predicate, params = location_predicate("s", query.location_type, query.location_id)
        time_sql, time_params = _time_predicate("e", "occurred_at", query)
        rows = self.repository.rows(f"""
          SELECT e.*,COALESCE(s.booth_id,s.market_id,s.field_id) location_id,
            CASE WHEN s.booth_id IS NOT NULL THEN 'booth' WHEN s.market_id IS NOT NULL THEN 'market' ELSE 'field' END location_type,
            COALESCE(s.booth_name,s.market_name,s.field_name) location_name
          FROM analytics_event e JOIN analytics_camera_source s USING(camera_id)
          WHERE e.occurred_at >= %s AND e.occurred_at < %s AND {predicate} AND {time_sql}
          ORDER BY e.occurred_at DESC LIMIT 100
        """, [start, end, *params, *time_params])
        return [{"id": row["event_id"], "title": row["title"], "severity": row["severity"],
                 "locationId": row["location_id"], "locationType": row["location_type"],
                 "locationName": row["location_name"], "occurredAt": row["occurred_at"].isoformat(),
                 "metricKey": row["metric_key"], "value": _number(row["value"])} for row in rows]


def _seconds_to_minutes(value: object) -> float | None:
    return round(float(value) / 60, 2) if value is not None else None


def _ratio(numerator: object, denominator: object, scale: float = 100) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(float(numerator) * scale / float(denominator), 2)


def _per_hour(value: object, start: datetime, end: datetime) -> float | None:
    if value is None:
        return None
    hours = max((end - start).total_seconds() / 3600, 1 / 60)
    return round(float(value) / hours, 2)


def _points(row: dict[str, object]) -> list[dict[str, float]]:
    value = row.get("points")
    return value if isinstance(value, list) else []


def _merge_spatial(rows: Iterable[dict[str, object]], layer: str) -> list[dict[str, float]]:
    cells: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["layer"] != layer:
            continue
        for point in _points(row):
            cells[(round(float(point["x"])), round(float(point["y"])))].append(float(point["value"]))
    if not cells:
        return []
    maximum = max((abs(sum(values) / len(values)) for values in cells.values()), default=1) or 1
    return [{"x": x, "y": y, "value": round(sum(values)/len(values), 3),
             "intensity": round(min(1, abs(sum(values)/len(values))/maximum), 4)} for (x, y), values in cells.items()]


def _difference_points(current: list[dict[str, float]], previous: list[dict[str, float]]) -> list[dict[str, float]]:
    first = {(int(item["x"]), int(item["y"])): item["value"] for item in current}
    second = {(int(item["x"]), int(item["y"])): item["value"] for item in previous}
    values = {key: first.get(key, 0)-second.get(key, 0) for key in first.keys() | second.keys()}
    maximum = max((abs(value) for value in values.values()), default=1) or 1
    return [{"x": key[0], "y": key[1], "value": round(value, 3), "intensity": round(value/maximum, 4)} for key, value in values.items()]


def _spatial_timeline(rows: list[dict[str, object]], layer: str) -> list[dict[str, object]]:
    buckets: dict[datetime, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["layer"] == layer:
            buckets[row["bucket_start"]].append(row)
    return [{"timestamp": timestamp.isoformat(), "points": _merge_spatial(items, layer)} for timestamp, items in sorted(buckets.items())[-336:]]


def _zone_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["zone_id"]), str(row["zone_name"]), str(row["location_id"]), str(row["location_type"]), str(row["location_name"]))
        groups[key].append(row)
    result = []
    for key, items in groups.items():
        averages = {layer: _average_point_value(items, layer) for layer in ("occupancy", "dwell", "traffic", "congestion")}
        coverages = [float(item["coverage_percent"]) for item in items if item.get("coverage_percent") is not None]
        result.append({"zoneId": key[0], "zoneName": key[1], "locationId": key[2], "locationType": key[3],
                       "locationName": key[4], "averageDwellMinutes": averages["dwell"], "p90DwellMinutes": None,
                       "occupancy": averages["occupancy"], "traffic": averages["traffic"],
                       "congestionPercent": averages["congestion"], "changePercent": None,
                       "coveragePercent": round(sum(coverages)/len(coverages), 2) if coverages else None})
    return result[:500]


def _average_point_value(rows: list[dict[str, object]], layer: str) -> float | None:
    values = [float(point["value"]) for row in rows if row["layer"] == layer for point in _points(row)]
    return round(sum(values)/len(values), 2) if values else None


def _average_dwell(zones: list[dict[str, object]]) -> float | None:
    values = [float(zone["averageDwellMinutes"]) for zone in zones if zone.get("averageDwellMinutes") is not None]
    return round(sum(values) / len(values), 2) if values else None


def _alerts(events: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{"id": event["id"], "severity": event["severity"], "title": event["title"],
             "locationName": event["locationName"], "occurredAt": event["occurredAt"]}
            for event in events if event["severity"] in ("critical", "warning")][:10]


def _queue_summary_card(queue_kpis: dict[str, object], queues: list[dict[str, object]]) -> dict[str, object]:
    active = [queue for queue in queues if (queue.get("currentLength") or 0) > 0]
    longest = max(active, key=lambda queue: queue.get("currentLength") or 0, default=None)
    return {
        "averageWaitMinutes": queue_kpis["averageWaitMinutes"],
        "slaPercent": queue_kpis["slaPercent"],
        "activeQueues": len(active),
        "longestQueueLocation": longest["locationName"] if longest else None,
    }


def _ranking(flow_locations: list[dict[str, object]], queue_locations: list[dict[str, object]]) -> list[dict[str, object]]:
    queue_by_id = {row["locationId"]: row for row in queue_locations}
    rows = [{
        "locationId": row["locationId"], "locationType": row["locationType"], "name": row["name"],
        "parentName": row["parentName"], "traffic": row["entries"], "occupancy": row["value"],
        "queueScore": queue_by_id.get(row["locationId"], {}).get("slaPercent"),
    } for row in flow_locations]
    rows.sort(key=lambda row: row["traffic"] if row["traffic"] is not None else -1, reverse=True)
    return rows[:20]


def _important_changes(flow_kpis: dict[str, object], queue_kpis: dict[str, object]) -> list[str]:
    candidates = [
        ("میانگین اشغال", flow_kpis["averageOccupancy"]["changePercent"]),
        ("ورودی‌ها", flow_kpis["entries"]["changePercent"]),
        ("میانگین زمان انتظار صف", queue_kpis["averageWaitMinutes"]["changePercent"]),
        ("درصد رعایت SLA صف", queue_kpis["slaPercent"]["changePercent"]),
    ]
    changes = []
    for label, change in candidates:
        if change is None or abs(change) < 5:
            continue
        direction = "افزایش" if change > 0 else "کاهش"
        changes.append(f"{label} {direction} {abs(change)}% نسبت به دوره قبل داشته است.")
    return changes


management_analytics_service = ManagementAnalyticsService()
