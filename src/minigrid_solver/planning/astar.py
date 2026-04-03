from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

from minigrid_solver.domain import AbstractState, PlanStep, SearchTrace
from minigrid_solver.planning.symbolic_model import PlannerModel


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

    def plan(self, initial_state: AbstractState) -> Tuple[List[PlanStep], SearchTrace]:
        self.trace = SearchTrace()
        self.trace.heuristic_at_start = self.heuristic(initial_state)
        self.trace.notes.append("Planning with A* over a factored state representation.")
        self.trace.notes.append(
            "The planner uses explicit preconditions/effects, similar to STRIPS operators from class."
        )
        self.trace.notes.append(f"Initial symbolic sub-goal: {PlannerModel.subgoal_description(initial_state)}.")

        frontier: List[Tuple[float, float, int, AbstractState]] = []
        heapq.heappush(frontier, (self.trace.heuristic_at_start, 0, 0, initial_state))

        came_from: Dict[AbstractState, Optional[AbstractState]] = {initial_state: None}
        action_from: Dict[AbstractState, Optional[PlanStep]] = {initial_state: None}
        g_cost: Dict[AbstractState, float] = {initial_state: 0.0}
        revisit_counts: Dict[Tuple[object, ...], int] = {PlannerModel.state_signature(initial_state): 1}
        counter = 0

        while frontier:
            _, cost_so_far, _, current = heapq.heappop(frontier)
            self.trace.expanded_nodes += 1

            if self.is_goal(current):
                plan = self._reconstruct_plan(current, came_from, action_from)
                self.trace.plan_cost = len(plan)
                return plan, self.trace

            for action, next_state, rationale in PlannerModel.successors(current):
                self.trace.generated_nodes += 1
                signature = PlannerModel.state_signature(next_state)
                revisit_penalty = 0.05 * revisit_counts.get(signature, 0)
                new_cost = cost_so_far + 1.0 + revisit_penalty
                if next_state not in g_cost or new_cost < g_cost[next_state]:
                    g_cost[next_state] = new_cost
                    revisit_counts[signature] = revisit_counts.get(signature, 0) + 1
                    counter += 1
                    priority = new_cost + self.heuristic(next_state)
                    heapq.heappush(frontier, (priority, new_cost, counter, next_state))
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
