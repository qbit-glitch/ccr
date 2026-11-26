import pytest

from ccr.core.plan import PlanStore, parse_plan, parse_usage


@pytest.fixture
def st(tmp_path):
    s = PlanStore(str(tmp_path / ".ccr"))
    s.set_goal("fix bug")
    return s


def test_hierarchy_brief_and_done(st):
    a, = st.add_tasks([{"title": "A", "acceptance": "t_a passes", "children": [{"title": "A1"}, {"title": "A2"}]}])
    assert "NEXT [T2]: A1" in st.brief()
    st.log("T2", "pass")
    assert "NEXT [T3]: A2" in st.brief() and "1/3" in st.brief()
    assert st.tree().splitlines()[1].startswith("  [T2] done")


def test_stuck_on_repeated_failures_and_packet(st):
    st.add_tasks([{"title": "A", "acceptance": "t passes"}])
    for i in range(3):
        st.log("T1", "fail", f"err{i}")
    assert st.stuck()[0]["reason"] == "repeated failures"
    assert "escalate" in st.brief()
    p = st.packet()
    assert "err2" in p and "t passes" in p and "NEED" in p


def test_stuck_on_flat_metric(st):
    st.add_tasks([{"title": "A"}])
    for m in (1, 2, 2, 2, 1):
        st.log("T1", "note", metric=m)
    assert st.stuck()[0]["reason"] == "no metric improvement"


def test_accept_plan_replaces_open_children_and_counts_escalation(st):
    st.add_tasks([{"title": "A", "children": [{"title": "done kid"}, {"title": "old kid", "children": [{"title": "gk"}]}]}])
    st.log("T2", "pass")
    st.log("T1", "fail", "x")
    st.accept_plan([{"title": "new kid"}], parent="T1")
    titles = {t["title"] for t in st._load()["tasks"].values()}
    assert titles == {"A", "done kid", "new kid"}


def test_budget_cap(st):
    st.set_budget(max_escalations=1)
    assert st.budget()["escalation_allowed"]
    assert "GOAL" in st.escalate()
    assert not st.budget()["escalation_allowed"]
    with pytest.raises(PermissionError):
        st.escalate()
    st.set_budget(max_tokens=10)
    st.add_tasks([{"title": "y"}])
    st.log("T1", "note", tokens=11)
    assert st.budget()["tokens"] == 11


def test_bad_inputs(st):
    with pytest.raises(KeyError):
        st.log("nope", "pass")
    with pytest.raises(ValueError):
        st.add_tasks([{"title": "a"}]) and st.log("T1", "weird")
    with pytest.raises(KeyError):
        st.add_tasks([{"title": "a"}], parent="zzz")


def test_parse_plan_from_pane_text():
    raw = 'thinking...\n```json\n[{"title": "a", "acceptance": "x"}]\n```\nDone.'
    assert parse_plan(raw)[0]["title"] == "a"
    with pytest.raises(ValueError):
        parse_plan("no plan here")
    with pytest.raises(ValueError):
        parse_plan('[{"nope": 1}]')


def test_mcp_plan_escalate(tmp_path):
    import ccr.mcp.server as srv
    from ccr.mcp import plan_tools

    srv._project_root = str(tmp_path)
    plan_tools.plan_update(goal="g", max_escalations=1)
    assert plan_tools.plan_escalate()["allowed"] is True
    assert plan_tools.plan_escalate()["allowed"] is False


def test_verify_runs_acceptance_command(st):
    st.add_tasks([{"title": "ok", "acceptance": "true"}, {"title": "bad", "acceptance": "echo boom; false"}, {"title": "none"}])
    assert st.verify("T1")["status"] == "done"
    assert st.verify("T2")["last_error"] == "boom"
    st.add_tasks([{"title": "color", "acceptance": "printf '\\033[31mred\\033[0m'; false"}])
    assert st.verify("T4")["last_error"] == "red"
    with pytest.raises(ValueError):
        st.verify("T3")


def test_cli_loop_end_to_end(tmp_path):
    """escalate -> accept raw pane text -> verify -> brief, via the real CLI."""
    from click.testing import CliRunner

    from ccr.cli_plan import plan

    (tmp_path / ".ccr").mkdir()
    PlanStore(str(tmp_path / ".ccr")).add_tasks([{"title": "fix", "acceptance": "false"}])
    run = lambda *a, **k: CliRunner().invoke(plan, ["--project", str(tmp_path), *a], **k)  # noqa: E731
    run("budget", "--max-escalations", "1")
    for _ in range(3):
        assert run("verify", "T1").exit_code == 1
    assert "stuck" in run("verify", "T1").output
    assert "STUCK [T1]" in run("escalate").output
    assert run("escalate").exit_code == 3
    pane = 'chatter\n[{"title": "fix2", "acceptance": "true"}]\nbye'
    assert run("accept", "-", "--parent", "T1", input=pane).exit_code == 0
    assert run("verify", "T2").exit_code == 0
    assert "PROGRESS: 2/2" in run("brief").output


OC = (
    'not json\n{"type":"text","part":{}}\n'
    '{"type":"step_finish","part":{"tokens":{"input":100,"output":10,"reasoning":5,"cache":{"read":2,"write":0}}}}\n'
    '{"type":"step_finish","part":{"tokens":{"input":200,"output":20}}}\n'
)


def test_parse_usage_and_local_tokens_do_not_hit_cap(st):
    assert parse_usage(OC) == 335 and parse_usage("garbage") == 0
    st.add_tasks([{"title": "a"}])
    st.set_budget(max_tokens=10)
    st.log("T1", "note", tokens=335, local=True)
    b = st.budget()
    assert b["local_tokens"] == 335 and b["tokens"] == 0 and b["escalation_allowed"]


def test_cli_ingest(tmp_path):
    from click.testing import CliRunner

    from ccr.cli_plan import plan

    (tmp_path / ".ccr").mkdir()
    PlanStore(str(tmp_path / ".ccr")).add_tasks([{"title": "a"}])
    r = CliRunner().invoke(plan, ["--project", str(tmp_path), "ingest", "T1", "-"], input=OC)
    assert r.output.strip() == "335"
    assert PlanStore(str(tmp_path / ".ccr")).budget()["local_tokens"] == 335


def test_failure_summary_is_one_line(st):
    st.add_tasks([{"title": "a"}])
    noisy = "\x1b[31m____ test ____\x1b[0m\n  foo = 1\nE   AssertionError: Lists differ\nE   + x\ntest.py:9: AssertionError\n1 failed"
    assert st.log("T1", "fail", noisy)["last_error"] == "E   AssertionError: Lists differ"
    assert "\n" not in st.brief().split("FAILS:")[1]


def test_parent_done_when_all_children_done(st):
    st.add_tasks([{"title": "P", "children": [{"title": "c1"}, {"title": "c2"}]}])
    st.log("T2", "pass")
    assert st._load()["tasks"]["T1"]["status"] != "done"
    st.log("T3", "pass")
    assert st._load()["tasks"]["T1"]["status"] == "done" and "3/3" in st.brief()


def test_replan_then_children_pass_completes_parent(st):
    st.add_tasks([{"title": "A", "acceptance": "x"}])
    st.accept_plan([{"title": "hint"}], parent="T1")
    st.log("T2", "pass")
    assert "2/2" in st.brief()
