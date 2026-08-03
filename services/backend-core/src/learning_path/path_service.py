"""PathService — core algorithm for personalized learning paths.

Computes optimal knowledge-point sequences by combining course graph
topology with user profile data (knowledge coverage, learning ability,
interaction style).
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from src.learning_path.models import (
    ContentType,
    ContentTypeSuggestion,
    KpStatus,
    PathNode,
    PathRecommendation,
    PersonalizedPath,
    ProgressRecord,
)

if TYPE_CHECKING:
    from src.kg.repositories.knowledge_point_repo import KnowledgePointRepository
    from src.kg.repositories.edge_repo import EdgeRepository

logger = logging.getLogger(__name__)

# ── Weight constants ──────────────────────────────────────────────────

W_READY = 0.5
W_DIFFICULTY = 0.3
W_GAP = 0.2

# Content type mapping by dominant interaction style
_STYLE_CONTENT_MAP: dict[str, list[ContentType]] = {
    "visual": ["visualization", "explanation"],
    "textual": ["explanation", "reading"],
    "interactive": ["exercise", "code"],
    "auditory": ["explanation"],
}


class PathService:
    """Computes personalized learning paths using KG + profile data.

    Injects::

        PathService(kp_repo, edge_repo)
    """

    def __init__(
        self,
        kp_repo: KnowledgePointRepository,
        edge_repo: EdgeRepository,
        db_pool=None,
    ) -> None:
        self._kp_repo = kp_repo
        self._edge_repo = edge_repo
        # In-memory progress store (prototype — replace with DB later)
        self._progress: dict[str, list[ProgressRecord]] = {}
        self._db_pool = db_pool  # Optional asyncpg pool for persistence
        self._progress_loaded: set[str] = set()

    async def _load_progress(self, user_id: str) -> None:
        """Load learning progress from PostgreSQL for a user."""
        if user_id in self._progress_loaded or self._db_pool is None:
            return

        self._progress_loaded.add(user_id)
        try:
            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT user_id, kp_id, status, score, time_spent_minutes, metadata,
                           created_at, updated_at
                    FROM learning_progress
                    WHERE user_id = $1
                    """,
                    user_id,
                )
                records = []
                for row in rows:
                    record = ProgressRecord(
                        user_id=row["user_id"],
                        course_id=row.get("kp_id", "").split("::")[0] if "::" in (row.get("kp_id") or "") else "",
                        kp_id=row["kp_id"],
                        status=row["status"],
                        quiz_score=row["score"],
                        completed_at=row["updated_at"].isoformat() if row["status"] == "completed" else None,
                    )
                    records.append(record)
                self._progress[user_id] = records
                logger.debug("Loaded %d progress records for user %s", len(records), user_id)
        except Exception as exc:
            logger.debug("Failed to load progress from DB: %s", exc)

    async def _save_progress(self, user_id: str, kp_id: str, status: str, score: float | None) -> None:
        """Persist learning progress to PostgreSQL."""
        if self._db_pool is None:
            return

        try:
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO learning_progress (user_id, kp_id, status, score)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (user_id, kp_id)
                    DO UPDATE SET
                        status = EXCLUDED.status,
                        score = COALESCE(EXCLUDED.score, learning_progress.score),
                        updated_at = NOW()
                    """,
                    user_id, kp_id, status, score,
                )
        except Exception as exc:
            logger.debug("Failed to save progress to DB: %s", exc)

    # ── Public API ──────────────────────────────────────────────────────

    async def get_personalized_path(
        self,
        course_id: str,
        user_id: str,
        profile: dict | None = None,
    ) -> PersonalizedPath:
        """Compute a full personalized learning path for a course."""
        # 1. Load course graph
        graph = await self._kp_repo.get_course_graph(course_id)
        if not graph.nodes:
            return PersonalizedPath(
                course_id=course_id,
                user_id=user_id,
                nodes=[],
                created_at=datetime.now(timezone.utc).isoformat(),
            )

        # 2. Topological sort
        sorted_ids = self._topo_sort(graph.nodes, graph.edges)

        # 3. Get profile status + progress
        await self._load_progress(user_id)
        mastery_map = self._extract_mastery(profile) if profile else {}
        user_progress = self._progress.get(user_id, [])
        progress_map = {r.kp_id: r for r in user_progress if r.course_id == course_id}

        # 4. Determine ability level
        ability = self._estimate_ability(profile)

        # 5. Build path nodes
        nodes: list[PathNode] = []
        mastered = 0
        completed = 0

        # Prerequisite lookup: kp_id → set of prerequisite ids
        prereq_map: dict[str, set[str]] = defaultdict(set)
        for edge in graph.edges:
            if edge.relation_type == "PREREQUISITE_OF":
                prereq_map[edge.target].add(edge.source)

        for kp_id in sorted_ids:
            kp = next((n for n in graph.nodes if n.id == kp_id), None)
            if not kp:
                continue

            # Determine status
            status, prereqs_met = self._determine_status(
                kp_id, prereq_map, mastery_map, progress_map,
            )

            if status == "completed":
                completed += 1
                if mastery_map.get(kp_id) == "mastered":
                    mastered += 1
            elif status == "mastered":
                mastered += 1

            nodes.append(
                PathNode(
                    kp_id=kp.id,
                    name=kp.name,
                    description=kp.description,
                    difficulty=kp.difficulty,
                    status=status,
                    prerequisites=list(prereq_map.get(kp_id, set())),
                    prerequisites_met=prereqs_met,
                    recommended_content_types=self._pick_content_types(profile),
                )
            )

        total = len(nodes)
        progress_pct = round((completed / total) * 100, 1) if total else 0.0

        return PersonalizedPath(
            course_id=course_id,
            user_id=user_id,
            nodes=nodes,
            total_count=total,
            mastered_count=mastered,
            completed_count=completed,
            progress_percent=progress_pct,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    async def get_next_recommendation(
        self,
        course_id: str,
        user_id: str,
        last_kp_id: str | None = None,
        profile: dict | None = None,
    ) -> PathRecommendation | None:
        """Find the next knowledge point the user should study."""
        path = await self.get_personalized_path(course_id, user_id, profile)
        if not path.nodes:
            return None

        # Score each ready node
        ability = self._estimate_ability(profile)
        mastery_map = self._extract_mastery(profile) if profile else {}
        gap_set = {
            kp_id for kp_id, status in mastery_map.items()
            if status == "learning"
        } if isinstance(mastery_map, dict) else set()
        if gap_set:
            logger.debug("Gap set (%d items): W_GAP=%.1f contributes to recommendation", len(gap_set), W_GAP)

        best_node: PathNode | None = None
        best_score = -1.0

        for node in path.nodes:
            if node.status not in ("ready", "in_progress"):
                continue
            if last_kp_id and node.kp_id == last_kp_id:
                continue

            score = self._priority_score(node, ability, gap_set)
            if score > best_score:
                best_score = score
                best_node = node

        if not best_node:
            return None

        content_type = self._pick_content_types(profile)
        session_min = (
            profile.get("focus_characteristics", {}).get("recommended_session_length", 20)
            if profile else 20
        )

        return PathRecommendation(
            next_kp_id=best_node.kp_id,
            next_kp_name=best_node.name,
            reason=self._recommendation_reason(best_node, ability, gap_set),
            recommended_content_type=content_type[0] if content_type else "explanation",
            estimated_session_min=session_min,
        )

    async def record_progress(
        self,
        user_id: str,
        course_id: str,
        kp_id: str,
        status: KpStatus,
        quiz_score: float | None = None,
    ) -> PersonalizedPath:
        """Record user progress and return the updated path."""
        if user_id not in self._progress:
            self._progress[user_id] = []

        # Remove old record if exists
        self._progress[user_id] = [
            r for r in self._progress[user_id]
            if not (r.course_id == course_id and r.kp_id == kp_id)
        ]

        record = ProgressRecord(
            user_id=user_id,
            course_id=course_id,
            kp_id=kp_id,
            status="completed" if status == "completed" else "in_progress",
            quiz_score=quiz_score,
            completed_at=datetime.now(timezone.utc).isoformat() if status == "completed" else None,
        )
        self._progress[user_id].append(record)

        # Persist to DB
        await self._save_progress(user_id, kp_id, record.status, quiz_score)

        logger.info("Progress recorded: user=%s course=%s kp=%s status=%s", user_id, course_id, kp_id, status)

        return await self.get_personalized_path(course_id, user_id)

    @staticmethod
    def get_content_type_suggestion(profile: dict | None) -> ContentTypeSuggestion:
        """Recommend content types based on interaction style."""
        if not profile:
            return ContentTypeSuggestion(content_types=["explanation"], dominant_style="textual", rationale="默认推荐")

        styles = profile.get("interaction_style", {})
        if not isinstance(styles, dict):
            return ContentTypeSuggestion(content_types=["explanation"], dominant_style="textual", rationale="默认推荐")

        best_style = max(styles, key=lambda k: float(styles.get(k, 0))) if styles else "textual"
        content_types = _STYLE_CONTENT_MAP.get(best_style, ["explanation"])

        # Map style to Chinese rationale
        style_labels = {
            "visual": "视觉型",
            "textual": "文字型",
            "interactive": "交互型",
            "auditory": "听觉型",
        }
        label = style_labels.get(best_style, "综合型")

        return ContentTypeSuggestion(
            content_types=content_types,
            dominant_style=best_style,
            rationale=f"根据您的{label}学习偏好，推荐：{'、'.join(content_types)}",
        )

    # ── Internal ────────────────────────────────────────────────────────

    @staticmethod
    def _topo_sort(nodes: list, edges: list) -> list[str]:
        """Kahn topological sort — returns node IDs in prerequisite order."""
        adj: dict[str, list[str]] = defaultdict(list)
        in_deg: dict[str, int] = defaultdict(int)
        all_ids = {n.id for n in nodes}

        for e in edges:
            if e.relation_type == "PREREQUISITE_OF" and e.source in all_ids and e.target in all_ids:
                adj[e.source].append(e.target)
                in_deg[e.target] += 1

        queue = deque(n for n in all_ids if in_deg.get(n, 0) == 0)
        result: list[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in adj.get(node, []):
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    queue.append(neighbor)

        # Add any remaining nodes not in DAG
        remaining = all_ids - set(result)
        result.extend(remaining)

        return result

    @staticmethod
    def _extract_mastery(profile: dict) -> dict[str, str]:
        """Extract KP mastery map from profile knowledge_coverage."""
        result: dict[str, str] = {}
        kc = profile.get("knowledge_coverage", {})
        if isinstance(kc, dict):
            for kp in kc.get("mastered", []):
                result[kp] = "mastered"
            for kp in kc.get("learning", []):
                result[str(kp)] = "learning"
        return result

    @staticmethod
    def _estimate_ability(profile: dict | None) -> float:
        """Estimate overall ability from profile (0-1 scale, inverted to 1-5 difficulty)."""
        if not profile:
            return 3.0  # default mid-level

        ability = profile.get("learning_ability", {})
        if not isinstance(ability, dict):
            return 3.0

        scores = [
            float(v) for v in ability.values()
            if isinstance(v, (int, float))
        ]
        if not scores:
            return 3.0

        avg = sum(scores) / len(scores)
        # Map 0-1 → 1-5 difficulty
        return 1.0 + avg * 4.0

    def _determine_status(
        self,
        kp_id: str,
        prereq_map: dict[str, set[str]],
        mastery_map: dict[str, str],
        progress_map: dict[str, ProgressRecord],
    ) -> tuple[KpStatus, bool]:
        """Determine a node's status based on mastery + progress."""
        # Check progress first
        if kp_id in progress_map:
            rec = progress_map[kp_id]
            if rec.status == "completed":
                return "completed", True
            return "in_progress", True

        # Check mastery
        mp = mastery_map.get(kp_id, "")
        if mp == "mastered":
            return "completed", True

        # Check prerequisites
        prereqs = prereq_map.get(kp_id, set())
        prereqs_met = True
        for prereq in prereqs:
            prereq_progress = progress_map.get(prereq)
            prereq_mastery = mastery_map.get(prereq, "")
            if prereq_progress and prereq_progress.status == "completed":
                continue
            if prereq_mastery == "mastered":
                continue
            prereqs_met = False
            break

        if prereqs_met:
            return "ready", True
        return "locked", False

    def _priority_score(
        self,
        node: PathNode,
        ability: float,
        gap_set: set[str],
    ) -> float:
        """Compute a priority score for a candidate node."""
        score = 0.0

        # Ready component (ready nodes get weight)
        if node.status in ("ready", "in_progress"):
            score += W_READY * 1.0

        # Difficulty match
        diff_diff = abs(node.difficulty - ability)
        if diff_diff <= 1.0:
            score += W_DIFFICULTY * 1.0
        elif diff_diff <= 2.0:
            score += W_DIFFICULTY * 0.5

        # Gap priority
        if node.kp_id in gap_set or node.name in gap_set:
            score += W_GAP * 1.0

        return score

    @staticmethod
    def _recommendation_reason(node: PathNode, ability: float, gap_set: set[str]) -> str:
        """Generate a human-readable reason for the recommendation."""
        parts = []
        if node.kp_id in gap_set or node.name in gap_set:
            parts.append("这是一个知识薄弱点")
        if node.prerequisites_met:
            parts.append("前置知识已掌握")
        diff_match = abs(node.difficulty - ability) <= 1.0
        if diff_match:
            parts.append("难度与您当前水平匹配")
        return "，".join(parts) if parts else "推荐继续学习"

    @staticmethod
    def _pick_content_types(profile: dict | None) -> list[ContentType]:
        """Pick content types based on interaction style."""
        suggestion = PathService.get_content_type_suggestion(profile)
        return suggestion.content_types
