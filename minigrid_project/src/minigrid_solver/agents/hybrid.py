from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from minigrid_solver.domain import AbstractState, PlanStep, PrimitiveAction, StepTrace
from minigrid_solver.perception import SymbolicStateExtractor
from minigrid_solver.planning import AStarPlanner


class ExplainableHybridAgent:
    """Planning-first agent with transparent re-planning traces."""

    name = "hybrid_astar"

    def __init__(self) -> None:
        self.extractor = SymbolicStateExtractor()
        self.planner = AStarPlanner()
        self.current_plan: List[PlanStep] = []
        self.reasoning_summary: List[str] = []
        self.plan_history: List[Dict[str, object]] = []
        self.total_expanded_nodes = 0
        self.total_generated_nodes = 0
        self.replan_count = 0

    def reset(self) -> None:
        self.current_plan = []
        self.reasoning_summary = []
        self.plan_history = []
        self.total_expanded_nodes = 0
        self.total_generated_nodes = 0
        self.replan_count = 0

    def _replan(self, state: AbstractState) -> None:
        self.replan_count += 1
        plan, trace = self.planner.plan(state)
        self.current_plan = plan
        self.total_expanded_nodes += trace.expanded_nodes
        self.total_generated_nodes += trace.generated_nodes
        current_subgoal = self.planner_subgoal(state)
        plan_payload = {
            "replan_index": self.replan_count,
            "symbolic_subgoal": current_subgoal,
            "symbolic_state": state.to_dict(),
            "search_trace": asdict(trace),
            "plan": [
                {
                    "action": step.action.name,
                    "rationale": step.rationale,
                    "predicted_state": step.predicted_state,
                }
                for step in plan
            ],
        }
        self.plan_history.append(plan_payload)

        if plan:
            self.reasoning_summary.append(
                f"Re-plan {self.replan_count}: sub-goal '{current_subgoal}', A* found a plan of length {len(plan)} after {trace.expanded_nodes} expansions."
            )
        else:
            self.reasoning_summary.append(
                f"Re-plan {self.replan_count}: sub-goal '{current_subgoal}', no plan found, so the controller will fall back to a safe exploratory action."
            )

    def planner_subgoal(self, state: AbstractState) -> str:
        from minigrid_solver.planning.symbolic_model import PlannerModel

        return PlannerModel.subgoal_description(state)

    def act(self, env, t: int) -> Tuple[int, Dict[str, object]]:
        state = self.extractor.extract(env)
        if not self.current_plan:
            self._replan(state)

        if self.current_plan:
            step = self.current_plan.pop(0)
            return int(step.action), {
                "policy_type": "symbolic_plan",
                "action_name": step.action.name,
                "rationale": step.rationale,
                "symbolic_subgoal": self.planner_subgoal(state),
                "symbolic_state": state.to_dict(),
            }

        fallback = PrimitiveAction.LEFT
        return int(fallback), {
            "policy_type": "fallback",
            "action_name": fallback.name,
            "rationale": "No valid symbolic plan was available, so the agent rotates to gather a different reachable state for the next re-plan.",
            "symbolic_subgoal": self.planner_subgoal(state),
            "symbolic_state": state.to_dict(),
        }

    def observe_transition(
        self,
        env,
        t: int,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        metadata: Dict[str, object],
    ) -> StepTrace:
        next_state = self.extractor.extract(env)
        self.current_plan = []
        return StepTrace(
            t=t,
            action=PrimitiveAction(action).name,
            rationale=str(metadata["rationale"]),
            symbolic_state=next_state.to_dict(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            metadata=metadata,
        )
