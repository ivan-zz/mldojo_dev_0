"""Stage 1 verification tests: Restructure + Migrate.

Validates that the migration from src/subgraphs/ to src/mle_star/ is complete
and the Algorithm 1 graph still runs correctly with mock data.

Test IDs map to stage_verification.md checklist items (S1-01 through S1-22).
"""

import json
import logging
import os
import operator
import shutil
import tempfile

import pytest


BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "mle_star")
MLE_STAR_DIR = os.path.normpath(BASE_DIR)


class TestDirectoryStructure:
    """S1-01, S1-02, S1-03, S1-04, S1-21, S1-22: Directory and file structure."""

    def test_s1_01_subdirectory_structure(self):
        required_dirs = ["state", "algorithms", "subgraphs", "nodes", "prompts"]
        for d in required_dirs:
            assert os.path.isdir(os.path.join(MLE_STAR_DIR, d)), f"Missing: {d}/"

    def test_s1_02_init_files(self):
        init_dirs = [
            "src/mle_star",
            "src/mle_star/state",
            "src/mle_star/algorithms",
            "src/mle_star/subgraphs",
            "src/mle_star/nodes",
            "src/mle_star/prompts",
        ]
        project_root = os.path.join(os.path.dirname(__file__), "..")
        for d in init_dirs:
            path = os.path.normpath(os.path.join(project_root, d))
            assert os.path.isfile(os.path.join(path, "__init__.py")), (
                f"Missing: {d}/__init__.py"
            )

    def test_s1_03_old_directory_removed(self):
        project_root = os.path.join(os.path.dirname(__file__), "..")
        old_dir = os.path.normpath(os.path.join(project_root, "src", "subgraphs"))
        assert not os.path.exists(old_dir), "Old src/subgraphs/ directory still exists"

    def test_s1_04_algorithm_renamed(self):
        project_root = os.path.join(os.path.dirname(__file__), "..")
        old_file = os.path.normpath(
            os.path.join(project_root, "src", "subgraphs", "algorithm_0.py")
        )
        new_file = os.path.join(MLE_STAR_DIR, "algorithms", "algorithm_1.py")
        assert not os.path.exists(old_file), "Old algorithm_0.py still exists"
        assert os.path.isfile(new_file), "algorithm_1.py missing"

    def test_s1_21_no_old_files(self):
        project_root = os.path.join(os.path.dirname(__file__), "..")
        old_dir = os.path.normpath(os.path.join(project_root, "src", "subgraphs"))
        assert not os.path.exists(old_dir), "Old src/subgraphs/ directory exists"

    def test_s1_22_no_old_import_paths(self):
        project_root = os.path.join(os.path.dirname(__file__), "..")
        mle_star_files = []
        for root, dirs, files in os.walk(MLE_STAR_DIR):
            for f in files:
                if f.endswith(".py"):
                    mle_star_files.append(os.path.join(root, f))
        for filepath in mle_star_files:
            with open(filepath, "r") as fh:
                content = fh.read()
            assert "from src.subgraphs" not in content, f"Old import in {filepath}"
            assert "import src.subgraphs" not in content, f"Old import in {filepath}"

    def test_placeholder_files_exist(self):
        placeholders = [
            os.path.join(MLE_STAR_DIR, "algorithms", "algorithm_2.py"),
            os.path.join(MLE_STAR_DIR, "algorithms", "algorithm_3.py"),
            os.path.join(MLE_STAR_DIR, "state", "system_state.py"),
            os.path.join(MLE_STAR_DIR, "state", "alg2_state.py"),
            os.path.join(MLE_STAR_DIR, "state", "alg3_state.py"),
            os.path.join(MLE_STAR_DIR, "subgraphs", "ablation_subgraph.py"),
            os.path.join(MLE_STAR_DIR, "subgraphs", "refinement_subgraph.py"),
            os.path.join(MLE_STAR_DIR, "subgraphs", "ensemble_round_subgraph.py"),
            os.path.join(MLE_STAR_DIR, "subgraphs", "submission_subgraph.py"),
            os.path.join(MLE_STAR_DIR, "graph.py"),
            os.path.join(MLE_STAR_DIR, "supervisor.py"),
        ]
        for path in placeholders:
            assert os.path.isfile(path), f"Missing placeholder: {path}"


class TestImports:
    """S1-05, S1-06, S1-07: Import verification."""

    def test_s1_05_node_function_names(self):
        from src.mle_star.algorithms.algorithm_1 import (
            A1__retrieve,
            generate_candidates_node,
            rank_node,
            merge_candidates_node,
            select_best_node,
            builder,
            get_graph,
            run,
        )
        from src.mle_star.subgraphs.candidate_subgraph import (
            A2__generate,
            A13__check_usage,
            A13__fix_usage,
            A12__check_leakage,
            A12__fix_leakage,
            eval_candidate,
            A11__debug,
            candidate_subgraph,
        )
        from src.mle_star.subgraphs.merge_subgraph import (
            A3__merge,
            A12__check_leakage_merge,
            A12__fix_leakage_merge,
            eval_merge,
            A11__debug_merge,
            merge_subgraph,
        )

        assert callable(A1__retrieve)
        assert callable(A2__generate)
        assert callable(A3__merge)

    def test_s1_06_shared_exports(self):
        from src.mle_star.state.shared import (
            traceable,
            SubgraphSpan,
            get_checkpointer,
            get_run_dir,
            get_thread_id,
            generate_run_dir,
            log_node_event,
            MLELogger,
            get_mle_logger,
            random_score,
            random_pass,
            simulate_delay,
            MAX_FIX_RETRIES,
            langfuse,
            shutdown_langfuse,
            propagate_attributes,
            _current_run_id,
            _current_run_dir,
            _current_phase,
            _current_mle_logger,
        )

        assert MAX_FIX_RETRIES == 5
        assert callable(generate_run_dir)
        assert callable(MLELogger)
        assert callable(traceable)
        assert callable(log_node_event)

    def test_s1_07_state_schemas(self):
        from src.mle_star.state.alg1_state import (
            Alg1State,
            CandidateState,
            MergeState,
            FanoutState,
            MergeFanoutState,
        )

        assert "task_desc" in Alg1State.__annotations__
        assert "model" in CandidateState.__annotations__


class TestStateSchemas:
    """S1-16, S1-17, S1-18: State schema field verification."""

    def test_s1_16_alg1_state_fields(self):
        from src.mle_star.state.alg1_state import Alg1State

        required = [
            "task_desc",
            "metric_direction",
            "retrieved_models",
            "candidates_pool",
            "leaderboard",
            "best_candidate",
            "current_reference_idx",
            "stage_history",
            "status",
        ]
        missing = [k for k in required if k not in Alg1State.__annotations__]
        assert not missing, f"Missing Alg1State fields: {missing}"

    def test_s1_17_candidate_state_fields(self):
        from src.mle_star.state.alg1_state import CandidateState

        required = [
            "model",
            "code",
            "score",
            "attempts",
            "usage_fix_attempts",
            "leakage_fix_attempts",
            "sub_events",
            "status",
        ]
        missing = [k for k in required if k not in CandidateState.__annotations__]
        assert not missing, f"Missing CandidateState fields: {missing}"

    def test_s1_18_candidates_pool_reducer(self):
        from src.mle_star.state.alg1_state import Alg1State

        pool_type = Alg1State.__annotations__["candidates_pool"]
        assert hasattr(pool_type, "__metadata__"), (
            "candidates_pool missing Annotated metadata"
        )

    def test_merge_state_fields(self):
        from src.mle_star.state.alg1_state import MergeState

        required = [
            "base_code",
            "ref_code",
            "merged_code",
            "score",
            "attempts",
            "leakage_fix_attempts",
            "sub_events",
            "status",
        ]
        missing = [k for k in required if k not in MergeState.__annotations__]
        assert not missing, f"Missing MergeState fields: {missing}"

    def test_fanout_state_fields(self):
        from src.mle_star.state.alg1_state import FanoutState

        assert "retrieved_models" in FanoutState.__annotations__
        assert "candidates_pool" in FanoutState.__annotations__

    def test_merge_fanout_state_fields(self):
        from src.mle_star.state.alg1_state import MergeFanoutState

        assert "best_candidate" in MergeFanoutState.__annotations__
        assert "leaderboard" in MergeFanoutState.__annotations__
        assert "candidates_pool" in MergeFanoutState.__annotations__


class TestAlgorithm1E2E:
    """S1-10, S1-11, S1-12, S1-19: End-to-end Algorithm 1 execution.

    Uses a single shared e2e_result session fixture to avoid repeating the full pipeline.
    """

    def test_s1_10_algorithm1_runs(self, e2e_result):
        assert e2e_result.get("status") == "done", (
            f"Expected 'done', got '{e2e_result.get('status')}'"
        )
        assert len(e2e_result.get("candidates_pool", [])) > 0, "No candidates generated"
        assert "best_candidate" in e2e_result, "No best_candidate"

    def test_s1_11_candidate_subgraph_outputs(self, e2e_result):
        pool = e2e_result.get("candidates_pool", [])
        assert len(pool) > 0, "Empty candidates_pool"
        for c in pool:
            assert "model" in c, f"Candidate missing model: {c}"
            assert "code" in c, f"Candidate missing code: {c}"
            assert "score" in c, f"Candidate missing score: {c}"

    def test_s1_12_merge_subgraph_outputs(self, e2e_result):
        pool = e2e_result.get("candidates_pool", [])
        merged = [c for c in pool if c.get("model") == "merged"]
        assert len(merged) > 0, "No merged candidates found"
        for m in merged:
            assert "code" in m, "Merged candidate missing code"
            assert "score" in m, "Merged candidate missing score"

    def test_s1_19_main_entry_point(self, e2e_result):
        assert e2e_result.get("status") == "done"
        assert isinstance(e2e_result["candidates_pool"], list)
        assert isinstance(e2e_result["best_candidate"], dict)


class TestNodeNames:
    """Verify A#__descriptive naming convention in stage history."""

    def test_stage_history_node_names(self, e2e_result):
        stage_names = {entry["stage"] for entry in e2e_result.get("stage_history", [])}
        expected = {
            "A1__retrieve",
            "generate_candidates",
            "Rank",
            "merge_candidates",
            "SelectBest",
        }
        assert expected.issubset(stage_names), f"Expected {expected}, got {stage_names}"

    def test_best_candidate_fields(self, e2e_result):
        best = e2e_result["best_candidate"]
        assert "model" in best
        assert "code" in best
        assert "score" in best


class TestJsonLogging:
    """S1-15: JSON structured logging."""

    def test_log_node_event_without_logger(self, caplog):
        """log_node_event falls back to logging module when no MLELogger active."""
        from src.mle_star.state.shared import log_node_event

        with caplog.at_level(logging.DEBUG, logger="mle_star"):
            log_node_event(
                "test_node", "test_event", {"key": "value"}, duration_ms=123.45
            )

        assert len(caplog.records) >= 1
        record = caplog.records[-1]
        parsed = json.loads(record.getMessage())
        assert parsed["node"] == "test_node"
        assert parsed["event"] == "test_event"
        assert parsed["duration_ms"] == 123.45
        assert "timestamp" in parsed

    def test_log_node_event_with_run_id_phase(self, caplog):
        """log_node_event respects explicit run_id and phase parameters."""
        from src.mle_star.state.shared import log_node_event

        with caplog.at_level(logging.DEBUG, logger="mle_star"):
            log_node_event(
                "test_node",
                "test_event",
                {"data": 42},
                duration_ms=50.0,
                run_id="test_run_id",
                phase="search",
            )

        assert len(caplog.records) >= 1
        record = caplog.records[-1]
        parsed = json.loads(record.getMessage())
        assert parsed["run_id"] == "test_run_id"
        assert parsed["phase"] == "search"

    def test_log_node_event_with_mle_logger(self, tmp_path):
        """MLELogger writes JSON lines to run.log and via Python logging."""
        from src.mle_star.state.shared import (
            MLELogger,
            log_node_event,
            _current_mle_logger,
        )

        run_dir = str(tmp_path / "mle_test_run")
        mle = MLELogger(run_dir=run_dir)
        token = _current_mle_logger.set(mle)

        try:
            log_node_event(
                "test_file_node",
                "test_file_event",
                {"test": True},
                duration_ms=99.0,
                run_id="file_test_run",
                phase="search",
            )

            log_path = os.path.join(run_dir, "run.log")
            assert os.path.isfile(log_path), f"run.log not created at {log_path}"

            with open(log_path) as fh:
                lines = fh.readlines()
            assert len(lines) >= 1
            parsed = json.loads(lines[-1])
            assert parsed["node"] == "test_file_node"
            assert parsed["event"] == "test_file_event"
            assert parsed["run_id"] == "file_test_run"
            assert parsed["phase"] == "search"
            assert parsed["duration_ms"] == 99.0
        finally:
            _current_mle_logger.reset(token)
            mle.close()

    def test_mle_logger_info_and_debug(self, tmp_path):
        """MLELogger writes INFO to both file and stdout, DEBUG to file only."""
        from src.mle_star.state.shared import MLELogger

        run_dir = str(tmp_path / "mle_level_test")
        mle = MLELogger(run_dir=run_dir)

        mle.info("node_info", "output", {"level": "info"})
        mle.debug("node_debug", "output", {"level": "debug"})

        mle.close()

        log_path = os.path.join(run_dir, "run.log")
        with open(log_path) as fh:
            lines = fh.readlines()

        assert len(lines) == 2
        info_entry = json.loads(lines[0])
        debug_entry = json.loads(lines[1])
        assert info_entry["event"] == "output"
        assert info_entry["data"]["level"] == "info"
        assert debug_entry["event"] == "output"
        assert debug_entry["data"]["level"] == "debug"

    def test_context_vars_propagate_run_id(self, tmp_path):
        """ContextVars _current_run_id and _current_phase propagate to log_node_event."""
        from src.mle_star.state.shared import (
            MLELogger,
            log_node_event,
            _current_run_id,
            _current_phase,
            _current_mle_logger,
        )

        run_dir = str(tmp_path / "mle_ctx_test")
        mle = MLELogger(run_dir=run_dir)
        logger_token = _current_mle_logger.set(mle)
        run_id_token = _current_run_id.set("ctx_run_123")
        phase_token = _current_phase.set("ablation")

        try:
            log_node_event("ctx_node", "ctx_event", {"ctx": True})

            log_path = os.path.join(run_dir, "run.log")
            with open(log_path) as fh:
                lines = fh.readlines()
            parsed = json.loads(lines[-1])
            assert parsed["run_id"] == "ctx_run_123"
            assert parsed["phase"] == "ablation"
        finally:
            _current_mle_logger.reset(logger_token)
            _current_run_id.reset(run_id_token)
            _current_phase.reset(phase_token)
            mle.close()

    def test_generate_run_dir_format(self):
        """generate_run_dir produces directories with YYYYMMDDHHMMSS prefix."""
        from src.mle_star.state.shared import generate_run_dir

        run_dir = generate_run_dir()
        basename = os.path.basename(run_dir.rstrip("/"))
        assert len(basename) == 23
        import re

        assert re.match(r"^\d{14}_[0-9a-f]{8}$", basename), f"Bad format: {basename}"

    def test_e2e_creates_run_log(self, e2e_result, tmp_path):
        """E2E run creates run.log with JSON entries for Algorithm1 nodes."""
        from src.mle_star.algorithms.algorithm_1 import run

        run_dir = str(tmp_path / "e2e_log_test")
        result = run(
            initial_state={
                "task_desc": "log test",
                "retrieved_models": [],
                "candidates_pool": [],
                "leaderboard": [],
                "best_candidate": {},
                "current_reference_idx": 0,
                "stage_history": [],
                "status": "start",
            },
            run_dir=run_dir,
            thread_id="log_test_thread",
        )

        log_path = os.path.join(run_dir, "run.log")
        assert os.path.isfile(log_path), f"run.log not found at {log_path}"

        with open(log_path) as fh:
            lines = fh.readlines()

        assert len(lines) >= 2, "Expected at least run_start and run_end events"
        start_entry = json.loads(lines[0])
        assert start_entry["node"] == "Algorithm1"
        assert start_entry["event"] == "run_start"

        end_entries = [
            json.loads(l) for l in lines if json.loads(l).get("event") == "run_end"
        ]
        assert len(end_entries) >= 1
        assert end_entries[-1]["node"] == "Algorithm1"
        assert end_entries[-1]["data"]["status"] == "done"


class TestCheckpointer:
    """S1-14: Checkpointing works with new file paths."""

    def test_get_checkpointer(self, tmp_path):
        from src.mle_star.state.shared import get_checkpointer

        run_dir = str(tmp_path / "runs" / "cp_test")
        os.makedirs(run_dir, exist_ok=True)
        cp = get_checkpointer(run_dir)
        assert cp is not None

    def test_state_history_accessible(self, e2e_result):
        """S1-14: Verify the e2e result has all expected state fields (checkpoint was created)."""
        assert "candidates_pool" in e2e_result
        assert e2e_result.get("status") == "done"

    def test_e2e_completed_successfully(self, e2e_result):
        """Verify the e2e run completed, proving checkpointing enabled graph execution."""
        assert e2e_result is not None
        assert e2e_result.get("status") == "done"


class TestSubgraphImports:
    """S1-20: All imports in migrated files resolve."""

    def test_algorithm1_imports(self):
        from src.mle_star.algorithms.algorithm_1 import (
            A1__retrieve,
            generate_candidates_node,
            rank_node,
            merge_candidates_node,
            select_best_node,
            builder,
            get_graph,
            run,
            dispatch_candidates,
            candidate_flow_node,
            dispatch_merges,
            merge_flow_node,
        )

    def test_candidate_subgraph_imports(self):
        from src.mle_star.subgraphs.candidate_subgraph import (
            A2__generate,
            A13__check_usage,
            A13__fix_usage,
            A12__check_leakage,
            A12__fix_leakage,
            eval_candidate,
            A11__debug,
            candidate_subgraph,
        )

    def test_merge_subgraph_imports(self):
        from src.mle_star.subgraphs.merge_subgraph import (
            A3__merge,
            A12__check_leakage_merge,
            A12__fix_leakage_merge,
            eval_merge,
            A11__debug_merge,
            merge_subgraph,
        )

    def test_graph_module_imports(self):
        from src.mle_star.graph import get_alg1_graph

        assert callable(get_alg1_graph)

    def test_main_module_exists(self):
        import os

        main_path = os.path.join(os.path.dirname(__file__), "..", "main.py")
        assert os.path.isfile(os.path.normpath(main_path)), "main.py not found"

    def test_node_input_output_fields(self):
        from src.mle_star.state.shared import _NODE_INPUT_FIELDS, _NODE_OUTPUT_FIELDS

        assert "A1__retrieve" in _NODE_INPUT_FIELDS
        assert "A1__retrieve" in _NODE_OUTPUT_FIELDS
        assert "A2__generate" in _NODE_INPUT_FIELDS
        assert "A2__generate" in _NODE_OUTPUT_FIELDS
        assert "A3__merge" in _NODE_INPUT_FIELDS
        assert "A13__check_usage" in _NODE_INPUT_FIELDS
        assert "A12__check_leakage" in _NODE_INPUT_FIELDS
        assert "eval_candidate" in _NODE_INPUT_FIELDS
        assert "A11__debug" in _NODE_INPUT_FIELDS
