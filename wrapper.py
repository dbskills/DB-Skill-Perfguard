#!/usr/bin/env python3
"""Perfguard Skill — single-file long-lived HTTP server wrapper.

Pairwise plan comparison optimizer for PostgreSQL. Generates alternative join
orders, compares each against the default plan using the PerfGuard pairwise
model, returns the best as a hinted query.

Design: lock-free optimize via an atomic (model, fg, mtime)
snapshot; batched pairwise scoring (parallel EXPLAIN + one model forward for
all candidates); background EXPLAIN ANALYZE for training data (separate pool);
training in a separate process (`wrapper.py --train --model-dir ...`) that
saves the model atomically; server picks up new weights via an mtime check.

Endpoints: GET /health, POST /optimize, GET /state, POST /shutdown.
"""

import argparse
import contextlib
import io
import json
import os
import pickle
import queue as queue_mod
import random
import signal
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

warnings.filterwarnings("ignore")

import torch
import joblib
import psycopg2
import sqlglot
import numpy as np

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_pth")
CANDIDATE_LIMIT_DEFAULT = 50
TRAIN_THRESHOLD_DEFAULT = 100
MAX_BUFFER_SIZE_DEFAULT = 10000
POOL_SIZE_DEFAULT = 4
# Safe-gate: candidates whose cost is more than this many times the default
# plan's cost are discarded before scoring (don't even consider a plan that
# would take minutes to execute), and background EXPLAIN ANALYZE / training
# is skipped when the chosen plan is catastrophically costlier than the
# default. The cold-start model occasionally selects such plans.
_MAX_COST_RATIO = 50.0


def _plan_total_cost(plan_json):
    """Total Cost from a PostgreSQL EXPLAIN (FORMAT JSON) plan, or None."""
    try:
        return float(plan_json[0]["Plan"]["Total Cost"])
    except Exception:
        return None


def _cost_ratio_too_high(chosen_cost, default_cost, max_ratio=_MAX_COST_RATIO):
    """True if the chosen plan's cost is catastrophically higher than the
    default's. Unknown / non-positive costs → False (let it through)."""
    if not chosen_cost or not default_cost:
        return False
    try:
        d = float(default_cost)
        if d <= 0:
            return False
        return float(chosen_cost) / d > max_ratio
    except (TypeError, ValueError, ZeroDivisionError):
        return False


# --------------------------------------------------------------------------- #
# DB connection pool.
# --------------------------------------------------------------------------- #
class ConnectionPool:
    def __init__(self, size=POOL_SIZE_DEFAULT):
        self.size = size
        self._pools = {}
        self._created = {}
        self._lock = threading.Lock()

    def _queue(self, dsn):
        with self._lock:
            q = self._pools.get(dsn)
            if q is None:
                q = queue_mod.Queue(maxsize=self.size)
                self._pools[dsn] = q
                self._created[dsn] = 0
            return q

    def get(self, dsn):
        q = self._queue(dsn)
        try:
            return q.get_nowait()
        except queue_mod.Empty:
            pass
        with self._lock:
            if self._created[dsn] < self.size:
                self._created[dsn] += 1
                conn = psycopg2.connect(dsn)
                conn.autocommit = True
                conn.set_client_encoding("UTF8")
                return conn
        return q.get(timeout=60)

    def put(self, dsn, conn):
        self._queue(dsn).put(conn)

    def discard(self, dsn, conn):
        self._queue(dsn)
        with self._lock:
            try:
                conn.close()
            except Exception:
                pass
            if self._created.get(dsn, 0) > 0:
                self._created[dsn] -= 1

    def close_all(self):
        with self._lock:
            for q in self._pools.values():
                while True:
                    try:
                        conn = q.get_nowait()
                    except queue_mod.Empty:
                        break
                    try:
                        conn.close()
                    except Exception:
                        pass
            self._pools.clear()
            self._created.clear()


# --------------------------------------------------------------------------- #
# SQL parsing (schema-agnostic).
# --------------------------------------------------------------------------- #
def extract_tables_and_join_edges(sql):
    parsed = sqlglot.parse_one(sql)
    tables = {}
    edges = []

    def _add_table(node):
        alias = node.alias_or_name.lower() if hasattr(node, 'alias_or_name') else str(node).lower()
        name = node.name.lower() if hasattr(node, 'name') else alias
        if alias and name:
            tables[alias] = name

    from_clause = parsed.find(sqlglot.exp.From)
    if from_clause:
        for tbl in from_clause.find_all(sqlglot.exp.Table):
            _add_table(tbl)
    for join in parsed.find_all(sqlglot.exp.Join):
        for tbl in join.find_all(sqlglot.exp.Table):
            _add_table(tbl)
        for sq in join.find_all(sqlglot.exp.Subquery):
            alias = sq.alias_or_name.lower() if hasattr(sq, 'alias_or_name') else None
            if alias:
                tables[alias] = alias
    for sq in parsed.find_all(sqlglot.exp.Subquery):
        if sq.alias_or_name:
            tables[sq.alias_or_name.lower()] = sq.alias_or_name.lower()
    for eq in parsed.find_all(sqlglot.exp.EQ):
        left_col, right_col = eq.left, eq.right
        if isinstance(left_col, sqlglot.exp.Column) and isinstance(right_col, sqlglot.exp.Column):
            l_alias = (left_col.table or "").lower()
            r_alias = (right_col.table or "").lower()
            if l_alias in tables and r_alias in tables and l_alias != r_alias:
                edges.append((l_alias, r_alias))
    return tables, edges


def generate_connected_join_orders(tables, edges, max_candidates=50):
    aliases = list(tables.keys())
    if len(aliases) <= 1:
        return []
    adj = {a: set() for a in aliases}
    for a1, a2 in edges:
        adj[a1].add(a2)
        adj[a2].add(a1)
    orders = []
    seen = set()
    attempts = 0
    max_attempts = max_candidates * 5
    while len(orders) < max_candidates and attempts < max_attempts:
        attempts += 1
        start = random.choice(aliases)
        order = [start]
        frontier = set(adj[start])
        remaining = set(aliases) - {start}
        while remaining:
            cands = [a for a in remaining if a in frontier]
            if not cands:
                break
            nxt = random.choice(cands)
            order.append(nxt)
            frontier |= adj[nxt]
            remaining.remove(nxt)
        if len(order) == len(aliases):
            t = tuple(order)
            if t not in seen:
                seen.add(t)
                orders.append(list(order))
    return orders


def get_plan_json(conn, query, hint=None, analyze=False, timeout_ms=300000):
    hint_prefix = f"/*+ Leading({hint}) */ " if hint else ""
    full_sql = (f"EXPLAIN (ANALYZE, FORMAT JSON) {hint_prefix}{query}" if analyze
                else f"EXPLAIN (FORMAT JSON) {hint_prefix}{query}")
    with conn.cursor() as cur:
        if analyze and timeout_ms:
            cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
        cur.execute(full_sql)
        raw = cur.fetchall()[0][0]
        return json.loads(raw) if isinstance(raw, str) else raw


def enable_hint_plan(conn):
    with conn.cursor() as cur:
        cur.execute("LOAD 'pg_hint_plan';")


def build_adjacency_matrix(plan_json, max_nodes):
    matrix = [[0] * max_nodes for _ in range(max_nodes)]
    node_idx = [0]

    def dfs(node, parent_idx):
        node_idx[0] += 1
        current_idx = node_idx[0] - 1
        if parent_idx >= 0:
            matrix[parent_idx][current_idx] = 1
            matrix[current_idx][parent_idx] = 1
        if "Plans" in node:
            for child in node["Plans"]:
                dfs(child, current_idx)

    dfs(plan_json, -1)
    return matrix


def _atomic_write(path, write_fn):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dir_ = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _model_files(model_dir):
    # Returns (nn_path, fg_path, dim_path, replay_path) for the model_dir.
    return (os.path.join(model_dir, "nn_weights"),
            os.path.join(model_dir, "feature_generator"),
            os.path.join(model_dir, "input_feature_dim"),
            os.path.join(model_dir, "replay_buffer.pkl"))


# --------------------------------------------------------------------------- #
# Training subprocess: load model + replay from disk, fine-tune, save atomically.
# --------------------------------------------------------------------------- #
def run_training(model_dir):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ImportConfig as _IC
    config = _IC.Config()
    config.CUDA = False
    config.device = torch.device("cpu")
    config.GPU_LIST = [0]
    config.model_name = model_dir
    _IC.config = config
    from perfguard import PerfGuard
    from feature_generation import feature, util
    from get_data import Get_Dataset_Test, _nn_path, _feature_generator_path, _input_feature_dim_path

    nn_path, fg_path, dim_path, replay_path = _model_files(model_dir)
    if not (os.path.exists(nn_path) and os.path.exists(fg_path) and os.path.exists(dim_path)):
        print(f"model not found at {model_dir}", file=sys.stderr)
        return 1
    test_dataset = Get_Dataset_Test({})
    model = test_dataset.load_model(model_dir)
    with open(dim_path, "rb") as f:
        input_dim = joblib.load(f)
    with open(fg_path, "rb") as f:
        feature_gen = joblib.load(f)
    if not os.path.exists(replay_path):
        return 0
    with open(replay_path, "rb") as f:
        buffer = pickle.load(f)
    entries = buffer.get("entries", [])
    if len(entries) < 2:
        return 0

    # Featurize each entry with the loaded feature_gen, then pad to the max
    # feature width across all entries. The model is (re)created to match that
    # max width AFTER it is known — the old code recreated using only
    # all_plans[0]'s width, so a mixed-width replay (different queries) fed a
    # max-width-padded batch into a narrower model → matmul shape error and
    # training never saved.
    x1_list, x2_list, labels, adj1_list, adj2_list = [], [], [], [], []
    for e in entries:
        try:
            # feature_gen.transform([plan]) returns (features, [latency]);
            # one latency per plan passed in. Capture both so we can label
            # candidate-better-than-default. (The old code did y[0][0] on a
            # single-plan transform, where y[0] is already the float latency
            # — TypeError 'float' object is not subscriptable — so every entry
            # was skipped and the model was never trained.)
            f1, y1 = feature_gen.transform([e["default_plan"]])
            f2, y2 = feature_gen.transform([e["candidate_plan"]])
            f1p = util.prepare_trees(f1, lambda x: x.get_feature(), lambda x: x.get_left(),
                                     lambda x: x.get_right(), cuda=False, device=None)
            f2p = util.prepare_trees(f2, lambda x: x.get_feature(), lambda x: x.get_left(),
                                     lambda x: x.get_right(), cuda=False, device=None)
            f1_arr = np.array(f1p); f2_arr = np.array(f2p)
            adj1 = np.array([build_adjacency_matrix(e["default_plan"][0]["Plan"], f1_arr.shape[1])])
            adj2 = np.array([build_adjacency_matrix(e["candidate_plan"][0]["Plan"], f2_arr.shape[1])])
            lat1 = y1[0] if y1 and y1[0] is not None else float('inf')
            lat2 = y2[0] if y2 and y2[0] is not None else float('inf')
            labels.append(1.0 if lat2 < lat1 else 0.0)
            x1_list.append(f1_arr); x2_list.append(f2_arr)
            adj1_list.append(adj1[0]); adj2_list.append(adj2[0])
        except Exception:
            continue
    if not labels:
        return 0

    max_nodes = max(x.shape[1] for x in x1_list + x2_list)
    max_feat = max(x.shape[2] for x in x1_list + x2_list)
    input_dim = int(max_feat)
    # (Re)create the model to match the padded input width.
    current_dim = None
    for name, param in model.named_parameters():
        if "gcn_layer1.linear.weight" in name:
            current_dim = param.shape[1]
            break
    if current_dim is None or current_dim != input_dim:
        model = PerfGuard(input_dim, config.embd_dim, config.tensor_dim, config.dropout).to(config.device)
        model.eval()
    x1p = np.zeros((len(x1_list), max_nodes, max_feat))
    x2p = np.zeros((len(x2_list), max_nodes, max_feat))
    for i, x in enumerate(x1_list):
        x1p[i, :x.shape[1], :x.shape[2]] = x
    for i, x in enumerate(x2_list):
        x2p[i, :x.shape[1], :x.shape[2]] = x
    a1p = np.zeros((len(adj1_list), max_nodes, max_nodes))
    a2p = np.zeros((len(adj2_list), max_nodes, max_nodes))
    for i, a in enumerate(adj1_list):
        a1p[i, :a.shape[0], :a.shape[1]] = a
    for i, a in enumerate(adj2_list):
        a2p[i, :a.shape[0], :a.shape[1]] = a
    labels_arr = np.array(labels, dtype=np.float32)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), config.init_lr)
    loss_fn = torch.nn.BCELoss()
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(min(config.epochs, 50)):
            output = model(a1p, a2p, x1p, x2p)
            target = torch.tensor(labels_arr).to(config.device)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    model.eval()

    # Save atomically: temp dir + os.replace per file.
    tmp_dir = f"{model_dir}.tmp.{os.getpid()}"
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        torch.save(model.state_dict(), os.path.join(tmp_dir, "nn_weights"))
        joblib.dump(feature_gen, os.path.join(tmp_dir, "feature_generator"))
        joblib.dump(input_dim, os.path.join(tmp_dir, "input_feature_dim"))
        for fn in os.listdir(tmp_dir):
            os.replace(os.path.join(tmp_dir, fn), os.path.join(model_dir, fn))
    except Exception as e:
        print(f"training save failed: {e}", file=sys.stderr)
        return 1
    finally:
        try:
            for fn in os.listdir(tmp_dir):
                p = os.path.join(tmp_dir, fn)
                if os.path.exists(p):
                    os.remove(p)
            os.rmdir(tmp_dir)
        except OSError:
            pass
    return 0


# --------------------------------------------------------------------------- #
# Skill: holds all in-memory state; one instance per server process.
# --------------------------------------------------------------------------- #
class PerfGuardSkill:
    def __init__(self):
        self.pool = ConnectionPool(POOL_SIZE_DEFAULT)
        self._bg_pool = ConnectionPool(max(1, POOL_SIZE_DEFAULT // 2))
        self._snapshot = (None, None, 0)  # (model, feature_gen, mtime)
        self.model_dir = None
        self.input_dim = None
        self.replay_buffer = {"entries": [], "trained_count": 0}
        self.candidate_limit = CANDIDATE_LIMIT_DEFAULT
        self.train_threshold = TRAIN_THRESHOLD_DEFAULT
        self.max_buffer_size = MAX_BUFFER_SIZE_DEFAULT
        self._state_lock = threading.Lock()
        self._training_in_progress = False
        self._train_proc = None  # Popen of the in-flight training subprocess
        self._shutting_down = False  # set by persist(); /optimize then 503s
        self._persist_lock = threading.Lock()  # single-flight: concurrent /shutdown runs persist once
        self._bg_queue = queue_mod.Queue()
        self._bg_thread = threading.Thread(target=self._bg_loop, daemon=True)
        self._bg_thread.start()
        # source module refs (imported lazily on first load)
        self._util = None
        self._feature = None
        self._PerfGuard = None
        self._Get_Dataset_Test = None
        self._config = None

    def _apply_config(self, config):
        self.candidate_limit = int(config.get("candidate_limit", CANDIDATE_LIMIT_DEFAULT))
        self.train_threshold = int(config.get("train_trigger", TRAIN_THRESHOLD_DEFAULT))
        self.max_buffer_size = int(config.get("replay_cap", MAX_BUFFER_SIZE_DEFAULT))
        pool_size = int(config.get("connection_pool_size", POOL_SIZE_DEFAULT))
        if self.pool.size != pool_size:
            self.pool.close_all()
            self.pool = ConnectionPool(pool_size)

    def ensure_loaded(self, config):
        model_dir = config.get("model_dir") or DEFAULT_MODEL_DIR
        self._apply_config(config)
        if self._snapshot[0] is None or model_dir != self.model_dir:
            self._load_initial(model_dir)

    def _load_initial(self, model_dir):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import ImportConfig as _IC
        cfg = _IC.Config()
        cfg.CUDA = False
        cfg.device = torch.device("cpu")
        cfg.GPU_LIST = [0]
        cfg.model_name = model_dir
        _IC.config = cfg
        self._config = cfg
        from perfguard import PerfGuard
        from feature_generation import feature, util
        from get_data import Get_Dataset_Test, _nn_path, _feature_generator_path, _input_feature_dim_path
        self._PerfGuard = PerfGuard
        self._feature = feature
        self._util = util
        self._Get_Dataset_Test = Get_Dataset_Test

        nn_path, fg_path, dim_path, replay_path = _model_files(model_dir)
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.replay_buffer = self._load_replay(replay_path)
        if not (os.path.exists(nn_path) and os.path.exists(fg_path) and os.path.exists(dim_path)):
            # Bootstrap: no pre-trained model. The first optimize() call will
            # fit a fresh feature generator on cost-only plans, init a random
            # model, and save it; online training then improves it.
            self._snapshot = (None, None, 0)
            return
        model = Get_Dataset_Test({}).load_model(model_dir)
        with open(dim_path, "rb") as f:
            input_dim = joblib.load(f)
        with open(fg_path, "rb") as f:
            feature_gen = joblib.load(f)
        mtime = os.path.getmtime(nn_path)
        self.model_dir = model_dir
        self.input_dim = input_dim
        self.replay_buffer = self._load_replay(replay_path)
        self._snapshot = (model, feature_gen, mtime)

    def _maybe_reload_model(self):
        if self.model_dir is None:
            return
        try:
            mtime = os.path.getmtime(os.path.join(self.model_dir, "nn_weights"))
        except OSError:
            return
        if mtime == self._snapshot[2]:
            return
        try:
            nn_path, fg_path, dim_path, _ = _model_files(self.model_dir)
            model = self._Get_Dataset_Test({}).load_model(self.model_dir)
            with open(dim_path, "rb") as f:
                input_dim = joblib.load(f)
            with open(fg_path, "rb") as f:
                feature_gen = joblib.load(f)
            self._snapshot = (model, feature_gen, mtime)
            self.input_dim = input_dim
        except Exception as e:
            print(f"model reload failed: {e}", file=sys.stderr)

    def _load_replay(self, replay_path):
        if os.path.exists(replay_path):
            try:
                with open(replay_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return {"entries": [], "trained_count": 0}

    def _save_replay(self):
        _atomic_write(os.path.join(self.model_dir, "replay_buffer.pkl"),
                      lambda f: pickle.dump(self.replay_buffer, f))

    def _explain(self, dsn, query, hint, analyze, pool):
        conn = pool.get(dsn)
        ok = False
        try:
            enable_hint_plan(conn)
            plan = get_plan_json(conn, query, hint=hint, analyze=analyze)
            ok = True
            return plan
        finally:
            if ok:
                try:
                    conn.rollback()
                except Exception:
                    pass
                pool.put(dsn, conn)
            else:
                pool.discard(dsn, conn)

    def _bootstrap_model(self, default_plan, cand_plans):
        """Fit a fresh feature generator on cost-only plans, init a random
        PerfGuard model, and save both atomically. Used when no pre-trained
        model exists; online training improves it afterwards."""
        plans_for_fit = [default_plan] + [p for p in cand_plans if p is not None]
        if not plans_for_fit:
            return None, None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fg = self._feature.FeatureGenerator()
                fg.fit(plans_for_fit)
            sample_feat, _ = fg.transform([plans_for_fit[0]])
            sample_prepared = self._util.prepare_trees(
                sample_feat, lambda x: x.get_feature(), lambda x: x.get_left(),
                lambda x: x.get_right(), cuda=False, device=None)
            input_dim = np.array(sample_prepared).shape[2]
            model = self._PerfGuard(input_dim, self._config.embd_dim,
                                    self._config.tensor_dim, self._config.dropout).to(self._config.device)
            model.eval()
            self.input_dim = input_dim
            # Atomic save: temp dir + os.replace per file.
            tmp_dir = f"{self.model_dir}.tmp.{os.getpid()}"
            os.makedirs(tmp_dir, exist_ok=True)
            try:
                torch.save(model.state_dict(), os.path.join(tmp_dir, "nn_weights"))
                joblib.dump(fg, os.path.join(tmp_dir, "feature_generator"))
                joblib.dump(input_dim, os.path.join(tmp_dir, "input_feature_dim"))
                for fn in os.listdir(tmp_dir):
                    os.replace(os.path.join(tmp_dir, fn), os.path.join(self.model_dir, fn))
            finally:
                try:
                    for fn in os.listdir(tmp_dir):
                        p = os.path.join(tmp_dir, fn)
                        if os.path.exists(p):
                            os.remove(p)
                    os.rmdir(tmp_dir)
                except OSError:
                    pass
            mtime = os.path.getmtime(os.path.join(self.model_dir, "nn_weights"))
            self._snapshot = (model, fg, mtime)
            return model, fg
        except Exception as e:
            print(f"bootstrap failed: {e}", file=sys.stderr)
            return None, None

    def _score_batch(self, fg, model, default_plan, cand_plans):
        """Batched pairwise scoring: one featurize + one model forward for all
        candidates vs the default. Returns (best_hint, best_score, num_valid)."""
        valid = [(i, p) for i, p in enumerate(cand_plans) if p is not None]
        if not valid:
            return None, -1.0, 0
        # Safe-gate: drop candidates whose cost is catastrophically higher
        # than the default plan's (don't consider plans that would take
        # minutes to execute / hang background training).
        default_cost = _plan_total_cost(default_plan)
        if default_cost and default_cost > 0:
            valid = [(i, p) for i, p in valid
                     if not _cost_ratio_too_high(_plan_total_cost(p), default_cost)]
            if not valid:
                return None, -1.0, 0
        # Featurize default + all candidates in one transform call.
        all_plans = [default_plan] + [p for _, p in valid]
        feats, _ = fg.transform(all_plans)
        prepared = self._util.prepare_trees(
            feats, lambda x: x.get_feature(), lambda x: x.get_left(),
            lambda x: x.get_right(), cuda=False, device=None)
        arrs = np.array(prepared)  # 3D: (n_plans, nodes, feat) — prepare_trees pads+combines
        max_nodes = arrs.shape[1]
        max_feat = arrs.shape[2]
        # Adjacency matrices: one per plan, padded to max_nodes (matches source
        # get_data.get_two_adjaceny_matrix: np.array([... for plan in X])).
        adjs = np.array([build_adjacency_matrix(all_plans[i][0]["Plan"], max_nodes)
                         for i in range(len(all_plans))])  # 3D: (n_plans, nodes, nodes)
        # Pair: default (index 0) vs each candidate.
        n = len(valid)
        x1 = np.zeros((n, max_nodes, max_feat))  # default repeated
        x2 = np.zeros((n, max_nodes, max_feat))  # candidates
        a1 = np.zeros((n, max_nodes, max_nodes))
        a2 = np.zeros((n, max_nodes, max_nodes))
        for j in range(n):
            x1[j] = arrs[0]
            a1[j] = adjs[0]
            x2[j] = arrs[1 + j]
            a2[j] = adjs[1 + j]
        with torch.no_grad():
            scores = model(a1, a2, x1, x2)
        best_hint = None
        best_adv = -1.0
        for j, (idx, _) in enumerate(valid):
            try:
                score = float(scores[j])  # >0.5: default faster; <0.5: candidate faster
            except Exception:
                continue
            adv = 1.0 - score  # candidate advantage
            if adv > best_adv:
                best_adv = adv
                best_hint = idx
        return best_hint, best_adv, len(valid)

    def optimize(self, dsn, query, optimize_only):
        self._maybe_reload_model()
        t0 = time.time()
        model, fg, _ = self._snapshot

        tables, edges = extract_tables_and_join_edges(query)
        aliases = list(tables.keys())
        if len(aliases) <= 1:
            return {"optimized_query": query,
                    "metadata": {"strategy_type": "pairwise-plan-comparison",
                                 "optimization_time": time.time() - t0, "estimated_impact": 0.0,
                                 "num_candidates": 0, "mode": "inference-only" if optimize_only else "online-training",
                                 "note": "single-table query, no join order optimization needed"}}

        candidate_orders = generate_connected_join_orders(tables, edges, self.candidate_limit)

        # Default plan + all candidate plans in parallel (cost-only EXPLAIN).
        conn = self.pool.get(dsn)
        try:
            enable_hint_plan(conn)
            default_plan = get_plan_json(conn, query, hint=None, analyze=False)
        finally:
            try:
                conn.rollback()
            except Exception:
                pass
            self.pool.put(dsn, conn)

        hints = [" ".join(o) for o in candidate_orders]
        cand_plans = [None] * len(hints)

        def _one(i):
            try:
                cand_plans[i] = self._explain(dsn, query, hints[i], False, self.pool)
            except Exception:
                cand_plans[i] = None

        if hints:
            with ThreadPoolExecutor(max_workers=max(1, self.pool.size)) as ex:
                list(ex.map(_one, range(len(hints))))

        # Bootstrap a fresh model if none is loaded yet (fit fg on the cost
        # plans collected above, init random weights, save). Online training
        # improves it afterwards.
        if model is None:
            model, fg = self._bootstrap_model(default_plan, cand_plans)
            self._snapshot = (model, fg, self._snapshot[2])

        best_idx, best_adv, num_valid = self._score_batch(fg, model, default_plan, cand_plans)
        best_hint = hints[best_idx] if best_idx is not None and best_idx < len(hints) else None
        optimized_query = f"/*+ Leading({best_hint}) */ {query}" if best_hint else query
        estimated_impact = best_adv * 100 if best_adv > 0 else 0.0
        mode = "inference-only" if optimize_only else "online-training"
        result = {"optimized_query": optimized_query,
                  "metadata": {"strategy_type": "pairwise-plan-comparison",
                               "optimization_time": time.time() - t0,
                               "estimated_impact": estimated_impact,
                               "num_candidates": len(hints), "mode": mode, "best_hint": best_hint}}

        if not optimize_only:
            # Safe-gate: skip background EXPLAIN ANALYZE / training if the
            # chosen candidate is catastrophically costlier than the default
            # (would hang the background thread on a doomed execution).
            chosen_cost = (_plan_total_cost(cand_plans[best_idx])
                           if best_idx is not None and best_idx < len(cand_plans)
                           and cand_plans[best_idx] is not None else None)
            default_cost = _plan_total_cost(default_plan)
            if not _cost_ratio_too_high(chosen_cost, default_cost):
                self._bg_queue.put(("collect", dsn, query, best_hint))
        return result

    def _bg_loop(self):
        while True:
            task = self._bg_queue.get()
            try:
                self._collect_training_data(task)
            except Exception as e:
                print(f"bg training-data collection failed: {e}", file=sys.stderr)

    def _collect_training_data(self, task):
        _, dsn, query, best_hint = task
        spawn = False
        try:
            default_exec = self._explain(dsn, query, None, True, self._bg_pool)
            if best_hint:
                best_exec = self._explain(dsn, query, best_hint, True, self._bg_pool)
            else:
                best_exec = default_exec
            default_latency = default_exec[0].get("Execution Time") if isinstance(default_exec, list) and default_exec else 0
            best_latency = best_exec[0].get("Execution Time") if isinstance(best_exec, list) and best_exec else 0
            with self._state_lock:
                self.replay_buffer["entries"].append({
                    "default_plan": default_exec,
                    "candidate_plan": best_exec,
                    "candidate_latency": best_latency or 0,
                })
                if len(self.replay_buffer["entries"]) > self.max_buffer_size:
                    removed = len(self.replay_buffer["entries"]) - self.max_buffer_size
                    self.replay_buffer["entries"] = self.replay_buffer["entries"][-self.max_buffer_size:]
                    self.replay_buffer["trained_count"] = max(0, self.replay_buffer["trained_count"] - removed)
                self._save_replay()
                untrained = len(self.replay_buffer["entries"]) - self.replay_buffer["trained_count"]
                if untrained >= self.train_threshold and not self._training_in_progress:
                    self._training_in_progress = True
                    self.replay_buffer["trained_count"] = len(self.replay_buffer["entries"])
                    spawn = True
        except psycopg2.OperationalError as e:
            # statement_timeout (QueryCanceled, pgcode 57014) or other transient
            # DB error during background EXPLAIN ANALYZE: skip this sample.
            print(f"[perfguard] background EXPLAIN ANALYZE "
                  f"{'timed out' if getattr(e, 'pgcode', None) == '57014' else 'failed'} "
                  f"(pgcode={getattr(e, 'pgcode', None)}); skipping training sample: {e}",
                  file=sys.stderr)
        except Exception as e:
            print(f"[perfguard] training-data collection failed: {e}", file=sys.stderr)
        if spawn:
            self._spawn_training_worker()

    def _spawn_training_worker(self):
        model_dir = self.model_dir

        def _run():
            try:
                p = subprocess.Popen(
                    [sys.executable, os.path.abspath(__file__), "--train", "--model-dir", model_dir])
                with self._state_lock:
                    self._train_proc = p
                p.wait()
            except Exception as e:
                print(f"training worker failed: {e}", file=sys.stderr)
            finally:
                with self._state_lock:
                    self._training_in_progress = False
                    self._train_proc = None

        threading.Thread(target=_run, daemon=True).start()

    def _wait_for_training(self, timeout=300):
        """Block until the in-flight training subprocess finishes (its model
        save completes) or the timeout expires."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._state_lock:
                in_progress = self._training_in_progress
            if not in_progress:
                return
            time.sleep(0.5)

    def persist(self):
        if not self._persist_lock.acquire(blocking=False):
            # Another persist() is in flight (concurrent /shutdown or SIGTERM
            # racing a skill_runner stop_wrapper); it will complete the save.
            # Returning here avoids two in-process run_training calls
            # clobbering the same pid-based temp dir.
            return
        try:
                # Don't bypass the model save on shutdown. The replay buffer holds
                # untrained entries in memory; save it first so the training worker /
                # restart can rehydrate. Then wait for any in-flight training subprocess
                # so its model save isn't abandoned, and if entries remain untrained,
                # run a final in-process training so the persisted model reflects all
                # collected data.
                self._shutting_down = True
                with self._state_lock:
                    untrained = len(self.replay_buffer["entries"]) - self.replay_buffer["trained_count"]
                    if self.model_dir is not None:
                        self._save_replay()
                self._wait_for_training(timeout=300)
                if self.model_dir is not None and untrained > 0:
                    try:
                        run_training(self.model_dir)
                    except Exception as e:
                        print(f"final training on shutdown failed: {e}", file=sys.stderr)
        finally:
            self._persist_lock.release()
            # Only the persist that acquired the lock shuts down the
            # server — a guarded-out persist (concurrent /shutdown)
            # must NOT start SERVER.shutdown, or daemon_threads=True
            # would let main() exit and kill this in-flight save.
            if SERVER is not None:
                threading.Thread(target=SERVER.shutdown, daemon=True).start()

    def state_summary(self):
        model = self._snapshot[0]
        with self._state_lock:
            return {"replay_buffer_len": len(self.replay_buffer["entries"]),
                    "trained_count": self.replay_buffer["trained_count"],
                    "model_loaded": model is not None, "model_dir": self.model_dir,
                    "pool_size": self.pool.size,
                    "training_in_progress": self._training_in_progress}


# --------------------------------------------------------------------------- #
# HTTP server.
# --------------------------------------------------------------------------- #
SERVER = None
SKILL = PerfGuardSkill()


def _drain_and_stop():
    try:
        SKILL.persist()  # persist() starts SERVER.shutdown in its finally (only if it owns the lock)
    except Exception as e:
        print(f"persist failed: {e}", file=sys.stderr)
        if SERVER is not None:
            threading.Thread(target=SERVER.shutdown, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send_json(200, {"status": "ok"})
        elif self.path.startswith("/state"):
            self._send_json(200, SKILL.state_summary())
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/optimize":
            if SKILL._shutting_down:
                self._send_json(503, {"error": "server is shutting down"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(n).decode() if n else "{}"
                req = json.loads(raw) if raw else {}
            except Exception as e:
                self._send_json(400, {"error": f"bad request body: {e}"})
                return
            dsn = req.get("dsn")
            query = req.get("query")
            optimize_only = bool(req.get("optimize_only"))
            cfg = req.get("config")
            if isinstance(cfg, str) and cfg:
                try:
                    cfg = json.loads(cfg)
                except Exception:
                    cfg = {"model_dir": cfg}
            cfg = cfg or {}
            if not isinstance(cfg, dict):
                self._send_json(400, {"error": "config must be a JSON object"})
                return
            if not dsn or not query:
                self._send_json(400, {"error": "dsn and query are required"})
                return
            try:
                SKILL.ensure_loaded(cfg)
                result = SKILL.optimize(dsn, query, optimize_only)
            except FileNotFoundError as e:
                self._send_json(400, {"error": str(e)})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            else:
                self._send_json(200, result)
        elif self.path == "/shutdown":
            self._send_json(200, {"status": "shutting down"})
            _drain_and_stop()
        else:
            self._send_json(404, {"error": "not found"})


def _signal_handler(signum, frame):
    _drain_and_stop()


def main():
    parser = argparse.ArgumentParser(description="Perfguard Query Optimizer server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "0")))
    parser.add_argument("--train", action="store_true", help="Run a single training cycle and exit (subprocess mode).")
    parser.add_argument("--model-dir", default=None, help="Model directory (for --train mode).")
    args = parser.parse_args()

    if args.train:
        sys.exit(run_training(args.model_dir or DEFAULT_MODEL_DIR))

    if not args.port:
        print("error: --port or PORT env required", file=sys.stderr)
        sys.exit(1)

    global SERVER
    SERVER = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    print(f"Perfguard skill server listening on 127.0.0.1:{args.port}", flush=True)
    try:
        SERVER.serve_forever()
    finally:
        SKILL.pool.close_all()
        SKILL._bg_pool.close_all()


if __name__ == "__main__":
    main()
