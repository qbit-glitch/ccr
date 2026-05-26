"""Enterprise gateway policy primitives.

This is a local policy gate, not a network service. It lets orgs place a
governance check between an agent and approved memory operations while keeping
CCR local-first.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from typing import Any

from ccr.core.governance import load_policy, save_policy, scan_text


@dataclass
class GatewayDecision:
    allowed: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reasons": self.reasons}


class EnterprisePolicyGateway:
    def __init__(self, project: str):
        self.project = os.path.abspath(project)
        self.ccr_root = os.path.join(self.project, ".ccr")
        self.policy = load_policy(self.ccr_root)

    def init_policy(self) -> str:
        if not self.policy.project_boundary:
            self.policy.project_boundary = self.project
        return save_policy(self.ccr_root, self.policy)

    def check_tool(self, tool_name: str, actor_role: str = "writer") -> GatewayDecision:
        reasons: list[str] = []
        allowed_tools = self.policy.approved_tools or []
        if allowed_tools and not any(fnmatch.fnmatch(tool_name, pat) for pat in allowed_tools):
            reasons.append(f"tool {tool_name} is not approved")
        grants = self.policy.roles.get(actor_role, [])
        if "*" not in grants and not grants:
            reasons.append(f"role {actor_role} has no permissions")
        return GatewayDecision(not reasons, reasons)

    def check_memory_write(self, text: str, actor_role: str = "writer") -> GatewayDecision:
        decision = self.check_tool("gcc_commit", actor_role=actor_role)
        reasons = list(decision.reasons)
        findings = scan_text(text)
        high = [f for f in findings if f.severity == "high"]
        if high:
            reasons.append(f"{len(high)} high-severity secret finding(s) in proposed memory")
        return GatewayDecision(not reasons, reasons)
