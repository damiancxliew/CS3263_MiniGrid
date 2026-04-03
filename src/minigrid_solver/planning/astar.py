from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from minigrid_solver.domain import AbstractState, PlanStep, SearchTrace
from minigrid_solver.planning.symbolic_model import PlannerModel


@dataclass(order=True)
class SearchNode:
    priority: float
    path_cost: float
    tie_breaker: int
    state: AbstractState = field(compare=False)


class AStarPlanner:
    """Forward state-space A* planner over the symbolic MiniGrid state."""

    def __init__(self) -> None:
        self.trace = SearchTrace()

    def heuristic(self, state: AbstractState) -> int:
        ax, ay = state.agent_pos
        gx, gy = state.goal_pos
        goal_dist = abs(ax - gx) + abs(ay - gy)

        locked_doors = [door for door in state.doors if door.is_locked and not door.is_open]
        if not locked_doors:
            return goal_dist

        if state.carrying_key is None and state.keys:
            nearest_key = min(abs(ax - key.pos[0]) + abs(ay - key.pos[1]) for key in state.keys)
            return min(goal_dist + 2, nearest_key + goal_dist)

        return goal_dist + 1

    @staticmethod
    def is_goal(state: AbstractState) -> bool:
        return state.agent_pos == state.goal_pos

    def action_cost(self, current: AbstractState, next_state: AbstractState) -> float:
        signature = PlannerModel.state_signature(next_state)
        revisit_penalty = 0.05 * self.revisit_counts.get(signature, 0)
        return 1.0 + revisit_penalty

    def plan(self, initial_state: AbstractState) -> Tuple[List[PlanStep], SearchTrace]:
        self.trace = SearchTrace()
        self.trace.heuristic_at_start = self.heuristic(initial_state)
        self.trace.notes.append("Planning with A* over a factored state representation.")
        self.trace.notes.append(
            "The planner uses explicit preconditions/effects, similar to STRIPS operators from class."
        )
        self.trace.notes.append(f"Initial symbolic sub-goal: {PlannerModel.subgoal_description(initial_state)}.")

        self.revisit_counts: Dict[Tuple[object, ...], int] = {PlannerModel.state_signature(initial_state): 1}
        frontier: List[SearchNode] = []
        counter = 0
        heapq.heappush(
            frontier,
            SearchNode(
                priority=float(self.trace.heuristic_at_start or 0),
                path_cost=0.0,
                tie_breaker=counter,
                state=initial_state,
            ),
        )

        came_from: Dict[AbstractState, Optional[AbstractState]] = {initial_state: None}
        action_from: Dict[AbstractState, Optional[PlanStep]] = {initial_state: None}
        best_cost_by_signature: Dict[Tuple[object, ...], float] = {PlannerModel.state_signature(initial_state): 0.0}

        while frontier:
            node = heapq.heappop(frontier)
            current = node.state
            cost_so_far = node.path_cost
            signature = PlannerModel.state_signature(current)

            if cost_so_far > best_cost_by_signature.get(signature, float("inf")):
                continue

            self.trace.expanded_nodes += 1

            if self.is_goal(current):
                plan = self._reconstruct_plan(current, came_from, action_from)
                self.trace.plan_cost = len(plan)
                return plan, self.trace

            for action, next_state, rationale in PlannerModel.successors(current):
                self.trace.generated_nodes += 1
                next_signature = PlannerModel.state_signature(next_state)
                new_cost = cost_so_far + self.action_cost(current, next_state)
                if new_cost < best_cost_by_signature.get(next_signature, float("inf")):
                    best_cost_by_signature[next_signature] = new_cost
                    self.revisit_counts[next_signature] = self.revisit_counts.get(next_signature, 0) + 1
                    counter += 1
                    priority = new_cost + self.heuristic(next_state)
                    heapq.heappush(
                        frontier,
                        SearchNode(
                            priority=priority,
                            path_cost=new_cost,
                            tie_breaker=counter,
                            state=next_state,
                        ),
                    )
                    came_from[next_state] = current
                    action_from[next_state] = PlanStep(
                        action=action,
                        rationale=rationale,
                        predicted_state=next_state.to_dict(),
                    )

        self.trace.notes.append("No symbolic plan reached the goal.")
        return [], self.trace

    @staticmethod
    def _reconstruct_plan(
        goal_state: AbstractState,
        came_from: Dict[AbstractState, Optional[AbstractState]],
        action_from: Dict[AbstractState, Optional[PlanStep]],
    ) -> List[PlanStep]:
        plan: List[PlanStep] = []
        current = goal_state
        while came_from[current] is not None:
            step = action_from[current]
            if step is None:
                break
            plan.append(step)
            parent = came_from[current]
            if parent is None:
                break
            current = parent
        plan.reverse()
        return plan
