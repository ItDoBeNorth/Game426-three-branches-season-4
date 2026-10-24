"""A Days at Three Branches agent built entirely from ``sandbox.village``."""

import heapq
import math
import re

from sandbox.observation_types import ThreeBranchesAction, ThreeBranchesObservation
from sandbox.village import action, day, geometry, layout, me, people, props

# Duration (in ticks) to hold "use" on a prop. Plot and shrine are well under their catalog
# durations; pump, bell, toggles and occupancy are about 1.5x the Season 2 holds. "sleep" is
# reserved for the end-of-day return home, not idle flavor.
_TIMED_USE_TICKS = {"plot": 50, "shrine": 75, "bell": 15, "pump": 5}
_TOGGLE_USE_TICKS = 5
_OCCUPANCY_USE_TICKS = (6, 9) 
_TOGGLE_TYPES = ("lantern", "hearth", "stall")
_OCCUPANCY_TYPES = ("bench", "repair_bench")
# Catalog durations: a timed prop stays done this long after its last use, then reverts
_REVERT_TICKS = {"plot": 600, "shrine": 300, "pump": 10, "bell": 40}
_OCCUPANCY_MEMORY_TICKS = 30  # a bench seen occupied is worth checking again after this long
_ARRIVAL_LOOK_TICKS = 3  # ticks spent turning to look at a target before giving up on it
# Catalog footprints (width, height) in cells, counted from a prop's cell corner
_PROP_SIZES = {
    "stall": (2, 2),
    "lantern": (1, 1),
    "bench": (2, 1),
    "shrine": (3, 3),
    "board": (2, 2),
    "plot": (4, 2),
    "hearth": (1, 1),
    "repair_bench": (2, 1),
    "pump": (1, 1),
    "bell": (1, 1),
}
_REPOSITION_TURNS = (90.0, -90.0, 180.0)  # tried in order when "use" would pick a different prop
_ARRIVAL_SLACK = 0.75  # how close to a repositioned standing spot counts as arrived
_CLAIM_TTL_TICKS = 150  # how long a heard "I'll take care of X" keeps X off our list
_HEADING_MATCH_DEGREES = 15.0  # someone walking within this angle of a prop is treated as heading there
_CLAIM_TEXT = "I'll take care of {}."
_CLAIM_PATTERN = re.compile(r"I'll take care of ([a-z0-9_]+)\.")
_IDLE_EMOTES = tuple(emote for emote in action.EMOTES if emote != "sleep")
# Each prop's untouched state. Anything else means another villager already handled it (a tended
# plot, a lit lantern) or is on it right now (an occupied bench), so there is no work left here.
_PROP_START_STATE = {
    "stall": "closed",
    "lantern": "unlit",
    "bench": "empty",
    "shrine": "untended",
    "plot": "overgrown",
    "hearth": "unlit",
    "repair_bench": "idle",
    "pump": "idle",
    "bell": "silent",
}
_STUCK_TICKS_LIMIT = 30  # abandon a destination after this many ticks of near-zero movement
_INTERACTION_COOLDOWN_TICKS = 300  # per pair; a greeting and its reply both restart it
_MIDDAY_TICKS = (481, 720)  # rules.json's midday phase; day/night is off this season, so go by tick
_FAMILY_TIME_TICKS = 40  # time spent together once family has met at the home doorway
_FAMILY_WAIT_TICKS = 120  # how long to wait at the doorway for family before going back to work
_FAMILY_TRAVEL_TICKS = 300  # give up the walk home if it takes longer than this
_FAMILY_MEET_DISTANCE = 3.0  # metres; this close counts as together
_VISITOR_NEWS_TICKS = 300  # after meeting the visitor, greetings this soon mention it
_VISITOR_NEWS = " I saw the visitor recently."
_REPLY_DELAY_TICKS = 3  # beat before answering someone who greeted us
_REPLY_WINDOW_TICKS = 30  # a greeting not answered within this long is let go
_GREETING_NOTICE_RANGE = 6.0  # metres (hearing range); a wave or nod from farther isn't meant for us
_GREETING_FACING_DEGREES = 30.0  # they must be facing us this closely for a wave or nod to count
_DISTANT_WAVE_COOLDOWN_TICKS = 40  # after a silent wave at someone out of earshot, don't re-notice them
_FAMILY_GREET_AFTER_TICK = 50  # family just left the same house, so they ignore each other until then
_GREETING_LINES = {
    "visitor": "Hello visitor.",
    "stranger": "Hello stranger.",
    "acquaintance": "Hello again.",
    "friend": "Hello friend.",
    "family": "How are you, family?",
}
_REPLY_LINES = {
    "visitor": "Have a good day.",
    "stranger": "Hey.",
    "acquaintance": "Hello indeed.",
    "friend": "Nice day, isn't it?",
    "family": "I'm good, I will see you later.",
}
_WAVE_RELATIONS = ("visitor", "friend", "family")  # everyone else gets a nod
_STARTLE_COOLDOWN_TICKS = 20  # separate & shorter, so startling doesn't use up the greet cooldown
_GREET_DELAY_TICKS = 10  # beat spent facing someone after noticing them, before actually greeting
_VISITOR_APPROACH_RANGE = 8.0  # blocks; only detour to greet if visitor is this close when noticed
_VISITOR_GREET_DISTANCE = 2.0  # blocks; walk to this distance before actually greeting
_BELL_RANGE = 50.0  # blocks; how far a villager will detour to answer a ringing bell
_RETURN_HOME_TICK = 1000  # of the 1200-tick day; when villagers head home for the "night"

_HOP_REACH_MIN = 5.0  # blocks; short legs toward a destination, sweeping between each
_HOP_REACH_MAX = 30.0
_OCCUPIED_CHECK_RANGE = 2.0  # blocks; someone this close to a prop already has it, pick another

_BREAK_MIN_TICKS = 15  # after completing a task, idle-hop for a random duration before next task
_BREAK_MAX_TICKS = 45

_LIGHT_TYPES = ("lantern", "hearth")

# Which prop types each role deliberately seeks out, once no one else is already on the nearest
# one. Only farmers touch plots; the building-wanderer role has no props of its own (see
# _advance_building) so it is intentionally left out of this mapping.
_ROLE_TARGET_TYPES = {
    "farmer": ("plot",),
    "shrine_keeper": ("shrine",),
    "wellkeeper": ("pump",),
    "stall_tender": ("stall",),
    "occupancy_tender": ("bench", "repair_bench"),
    "lighter": _LIGHT_TYPES,
}

# 10 seats this season: 2 farmers, 2 shrine keepers, 1 wellkeeper, 2 stall tenders, 1 occupancy
# tender, 1 lighter, 1 building-wanderer. Indexed by player number (wraps if the cast is smaller).
_ROLE_ORDER = (
    "farmer",
    "farmer",
    "shrine_keeper",
    "shrine_keeper",
    "wellkeeper",
    "stall_tender",
    "stall_tender",
    "occupancy_tender",
    "lighter",
    "building_wanderer",
)


def _cell_centre(cell: dict[str, int]) -> dict[str, float]:
    """Return the point at the centre of one village cell."""

    return {"x": cell["x"] + 0.5, "y": cell["y"] + 0.5}


def _player_number(player_id: str) -> int:
    return int(player_id.split("_")[1])


def _role_of(player_id: str) -> str:
    """Any villager's role, computed the same way in every instance."""
    return _ROLE_ORDER[(_player_number(player_id) - 1) % len(_ROLE_ORDER)]


def _prop_centre(prop) -> dict[str, float]:
    """Middle of a prop's footprint (its ``cell`` is the corner)."""
    width, height = _PROP_SIZES.get(prop["type"], (1, 1))
    return {"x": prop["cell"]["x"] + width / 2, "y": prop["cell"]["y"] + height / 2}


def _prop_untouched(observation, prop) -> bool:
    """Whether a prop still needs doing and nobody else is on it.

    ``props.usable`` selects by reach and line of sight alone, regardless of facing, so it can
    hand back a prop that is not actually in the vision cone. ``props.state`` needs the cone, so
    that case reads as unknown here -- and an unconfirmed reading is treated as "leave it alone"
    rather than "assume untouched", since using an already-lit lantern would toggle it back off.
    """
    start = _PROP_START_STATE.get(prop["type"])
    if start is None:
        return True  # e.g. "board", which has no state to speak of
    return props.state(observation, prop["id"]) == start


def _nearest_walkable_cell(observation, cell, max_radius=15):
    """Return the closest walkable cell to one that may itself be blocked (e.g. a prop's footprint)."""
    if layout.walkable(observation, cell):
        return cell
    for radius in range(1, max_radius + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                candidate = {"x": cell["x"] + dx, "y": cell["y"] + dy}
                if layout.walkable(observation, candidate):
                    return candidate
    return cell


_SQRT2 = math.sqrt(2.0)
_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
_SEARCH_BUDGET = 5000  # heap pops before a search gives up and the goal counts as unreachable
_HEURISTIC_WEIGHT = 2.0  # above 1 the search heads for the goal: paths a bit longer, far less work


def _octile(node, goal):
    """Straight-line cost across an 8-way grid; never more than the real path cost."""
    dx, dy = abs(node[0] - goal[0]), abs(node[1] - goal[1])
    return (dx + dy) + (_SQRT2 - 2.0) * min(dx, dy)


def _step_cost(observation, cost_cache, cell):
    """Time to cross one cell: slow ground (fields, reeds) costs more, so paths favour roads."""
    cost = cost_cache.get(cell)
    if cost is None:
        ground = layout.ground_at(observation, {"x": cell[0], "y": cell[1]})
        cost = 1.0 / max(layout.SPEED_LIMITS.get(ground, 1.0), 0.1)
        cost_cache[cell] = cost
    return cost


def _astar_path(observation, edge_cache, cost_cache, start_cell, dest_cell):
    """A good-enough 8-way path of cells from start to dest, or None when none is found cheaply.

    A diagonal step is only taken when both cells beside it are clear too, so a body never clips a
    wall corner. The search is weighted toward the goal and gives up after ``_SEARCH_BUDGET`` pops,
    so an unreachable goal costs a bounded amount. ``edge_cache`` remembers ``layout.can_step``
    answers and ``cost_cache`` the per-cell ground cost; neither changes during a day.
    """
    start = (int(start_cell["x"]), int(start_cell["y"]))
    goal = (int(dest_cell["x"]), int(dest_cell["y"]))
    if start == goal:
        return []

    def clear(first, second):
        key = (first, second)
        known = edge_cache.get(key)
        if known is None:
            known = layout.can_step(
                observation, {"x": first[0], "y": first[1]}, {"x": second[0], "y": second[1]}
            )
            edge_cache[key] = known
        return known

    best = {start: 0.0}
    parent = {}
    closed = set()
    queue = [(0.0, 0.0, start)]
    pops = 0
    while queue:
        _, _, node = heapq.heappop(queue)
        if node in closed:
            continue
        pops += 1
        if pops > _SEARCH_BUDGET:
            return None
        if node == goal:
            path = []
            while node != start:
                path.append({"x": node[0], "y": node[1]})
                node = parent[node]
            path.reverse()
            return path
        closed.add(node)
        cost = best[node]
        for dx, dy in _STEPS:
            neighbour = (node[0] + dx, node[1] + dy)
            if neighbour in closed:
                continue
            if dx and dy:
                beside_x = (node[0] + dx, node[1])
                beside_y = (node[0], node[1] + dy)
                if not (
                    clear(node, beside_x)
                    and clear(node, beside_y)
                    and clear(beside_x, neighbour)
                    and clear(beside_y, neighbour)
                ):
                    continue
            elif not clear(node, neighbour):
                continue
            new_cost = cost + (_SQRT2 if dx and dy else 1.0) * _step_cost(observation, cost_cache, neighbour)
            if new_cost < best.get(neighbour, math.inf):
                best[neighbour] = new_cost
                parent[neighbour] = node
                heuristic = _octile(neighbour, goal)
                heapq.heappush(queue, (new_cost + _HEURISTIC_WEIGHT * heuristic, heuristic, neighbour))
    return None


class Agent:
    """Leaves home, does its role's rounds, wanders the village, and reacts to people it sees."""

    def reset(self, seed: int, observation: ThreeBranchesObservation) -> None:
        """Initialize per-day state: home, role, and tracking structures."""

        # Home & doorway (static for the day)
        self.home = me.home(observation)
        self.home_doorway = layout.doorway(observation, self.home) if self.home != "none" else None

        # Role: fixed by player number, not randomized. Determines which props (if any) this
        # villager deliberately seeks out all day; see _ROLE_TARGET_TYPES and _advance_building.
        index = int(me.player_id(observation).split("_")[1]) - 1
        self.role = _ROLE_ORDER[index % len(_ROLE_ORDER)]
        self.player_id = me.player_id(observation)
        self.number = index + 1
        self.reposition_tries = 0  # 90-degree moves around the current target so "use" picks it
        self.claims = {}  # {prop_id: (player_id, tick)} heard "I'll take care of" messages
        self.claim_told = set()  # villagers already told about the current target

        self.rng = me.rng(observation, seed)  # kept for hop picks, idle emotes, and building picks

        # Day-phase tracking
        self.phase = "task"  # "task" -> "returning"
        self.announced_role = False  # broadcast role once at start of day
        self.used_props = set()  # non-timed prop ids already used today, never repeated
        self.timed_done = {}  # {prop_id: tick} when this villager finished a timed prop
        self.prop_memory = {}  # {prop_id: (state, tick)} as last seen
        self.arrival_looks = 0  # ticks spent turning to see the current target on arrival
        self.interactions = {}  # {player_id: {"last", "count", "initiated_by"}} greetings and replies
        self.reply_due = {}  # {player_id: tick they greeted us} still waiting for our answer
        self.homes = {entry["id"]: entry["home"] for entry in people.roster(observation)}
        self.seen_ever = set()  # everyone this villager has laid eyes on today
        self.distant_waves = {}  # {player_id: tick} of silent waves at people out of earshot
        self.family_lunch = None  # {"stage": going|waiting|together, "since", "partner"?} during family time
        self.family_lunch_done = False  # once a day
        self.met_visitor_tick = None  # last real exchange with the visitor
        self.news_told = set()  # villagers already told about that visitor meeting
        self._passing_emote = None  # wave added to this tick's action after greeting a friend on the move
        self.startled = {}  # {player_id: last_startled_tick}, separate from greeted on purpose
        self.pending_greet = None  # {"id", "ready_tick", "approach_player"?} while greeting

        # Role-task state: the prop (or, for building_wanderer, doorway) currently claimed as
        # this villager's next job, and the short waypoint it is hopping toward on the way there
        self.role_target = None  # {"prop_id", "type", "position"} or None
        self.hop_point = None
        self.active_task = None  # {"prop_id": str, "ticks_remaining": int} while mid-use
        self.use_counter = 0
        self.break_ticks_remaining = 0  # idle-hop after completing a task before picking next

        # Direct-walk destination, used only for the bell response (no hops -- answer it quickly)
        self.current_destination = None

        # Pathfinding cache, shared by both the bell response and the hop travel above
        self._path = None  # cached cardinal-step path to whichever destination is active
        self._path_destination = None  # (x, y) tuple of cached path's target
        self.stuck_ticks = 0  # consecutive near-zero-movement ticks while walking a destination
        self._edge_cache = {}  # can_step answers for this day's village, shared by every path search
        self._cost_cache = {}  # per-cell ground cost, same lifetime
        self._failed_goals = set()  # goal cells a search gave up on; never searched again today
        self._walk_failed = False  # set by _walk_toward when the current destination has no route

        # Messages queued by act() for chat(), which the harness calls later the same tick
        self._outbox = []

    def _prop_use_ticks(self, prop_type: str) -> int:
        """How many ticks to hold "use" on a prop, based on its catalog transition type."""
        if prop_type in _TIMED_USE_TICKS:
            return _TIMED_USE_TICKS[prop_type]
        if prop_type in _TOGGLE_TYPES:
            return _TOGGLE_USE_TICKS
        if prop_type in _OCCUPANCY_TYPES:
            return self.rng.randint(*_OCCUPANCY_USE_TICKS)
        return 1  # e.g. "board": no state to hold, just a passing glance

    @property
    def is_performing(self) -> bool:
        """Mid-use on a prop right now."""
        return self.active_task is not None

    @property
    def is_chosen(self) -> bool:
        """Has picked a prop to work on but hasn't started using it yet."""
        return self.active_task is None and self.role_target is not None and self.role_target["prop_id"] is not None

    def _find(self, observation: ThreeBranchesObservation, start_cell, dest_cell):
        """Path between two cells using this day's caches, or None when the search gives up."""
        return _astar_path(observation, self._edge_cache, self._cost_cache, start_cell, dest_cell)

    def _walk_toward(self, destination: dict[str, float], observation: ThreeBranchesObservation) -> float:
        """Heading toward destination, following a cached path around obstacles."""
        here = me.position(observation)
        here_cell = layout.cell_at(observation, here)
        dest_cell = layout.cell_at(observation, destination)
        if here_cell is None or dest_cell is None:
            return geometry.heading_to(here, destination)
        dest_cell = _nearest_walkable_cell(observation, dest_cell)

        cache_key = (dest_cell["x"], dest_cell["y"])
        if self._path_destination != cache_key or self._path is None:
            path = None if cache_key in self._failed_goals else self._find(observation, here_cell, dest_cell)
            self._walk_failed = path is None
            if path is None:
                self._failed_goals.add(cache_key)
            self._path = path or []
            self._path_destination = cache_key

        # A diagonal step can carry us past the next cell, so drop everything up to the cell we stand in
        for index, cell in enumerate(self._path[:4]):
            if cell["x"] == here_cell["x"] and cell["y"] == here_cell["y"]:
                del self._path[: index + 1]
                break

        if not self._path:
            return geometry.heading_to(here, destination)
        return geometry.heading_to(here, _cell_centre(self._path[0]))

    def _random_nearby_point(self, observation: ThreeBranchesObservation, here) -> dict[str, float]:
        """A random walkable point a short hop away, for idling when a role has no work left."""
        for _ in range(20):
            angle = math.radians(self.rng.uniform(0.0, 360.0))
            reach = self.rng.uniform(_HOP_REACH_MIN, _HOP_REACH_MAX)
            point = {"x": here["x"] + reach * math.cos(angle), "y": here["y"] + reach * math.sin(angle)}
            cell = layout.cell_at(observation, point)
            if cell is None or not layout.walkable(observation, cell):
                continue
            if (cell["x"], cell["y"]) in self._failed_goals:
                continue  # a search already gave up on this spot, so pick somewhere else
            return _cell_centre(cell)
        return here

    def _pick_hop_point(self, observation: ThreeBranchesObservation, here, target_pos) -> dict[str, float]:
        """A short waypoint along the real path toward target_pos, so travel comes in small hops."""
        here_cell = layout.cell_at(observation, here)
        dest_cell = layout.cell_at(observation, target_pos)
        if here_cell is None or dest_cell is None:
            return target_pos
        dest_cell = _nearest_walkable_cell(observation, dest_cell)
        path = self._find(observation, here_cell, dest_cell) or []
        if not path:
            return target_pos
        hop_len = self.rng.uniform(_HOP_REACH_MIN, _HOP_REACH_MAX)
        travelled = 0.0
        previous = here
        for cell in path:
            centre = _cell_centre(cell)
            travelled += geometry.distance(previous, centre)
            previous = centre
            if travelled >= hop_len:
                return centre
        return _cell_centre(path[-1])

    def _prop_needs_doing(self, prop, tick: int) -> bool:
        """Best guess whether a prop still needs doing, from what this villager has done and seen."""
        prop_id, prop_type = prop["id"], prop["type"]
        if prop_id in self.used_props:
            return False
        revert = _REVERT_TICKS.get(prop_type)
        if revert is not None and tick - self.timed_done.get(prop_id, -revert) < revert:
            return False  # we tended it and it hasn't reverted yet
        start = _PROP_START_STATE.get(prop_type)
        remembered = self.prop_memory.get(prop_id)
        if start is None or remembered is None:
            return True  # stateless (board) or never seen: worth a visit
        state, seen_tick = remembered
        if state == start:
            return True
        if revert is not None:
            return tick - seen_tick >= revert  # seen tended; reverts at the latest this long after
        if prop_type in _OCCUPANCY_TYPES:
            return tick - seen_tick >= _OCCUPANCY_MEMORY_TICKS
        return False  # a lit lantern or open stall stays done

    def _available_props(self, observation: ThreeBranchesObservation, prop_types):
        """Every prop of these types, anywhere in the village, that seems to need doing and that
        nobody visible is already standing at."""
        tick = day.tick(observation)
        here = me.position(observation)
        watchers = [person["position"] for person in people.seen(observation)]
        watchers += [person["position"] for person in people.nearby(observation)]
        candidates = []
        for prop in props.all(observation):
            if prop["type"] not in prop_types:
                continue
            if not self._prop_needs_doing(prop, tick):
                continue
            if self._someone_else_has(observation, prop, here, tick):
                continue
            prop_pos = _cell_centre(prop["cell"])
            if any(geometry.distance(prop_pos, watcher) <= _OCCUPIED_CHECK_RANGE for watcher in watchers):
                continue  # someone is already standing right on this one -- let them have it
            candidates.append(prop)
        return candidates

    def _random_available_prop(self, observation: ThreeBranchesObservation, prop_types):
        """Random untended prop of these types that nobody visible is already working, or None."""
        candidates = self._available_props(observation, prop_types)
        if not candidates:
            return None
        return self.rng.choice(candidates)

    def _opportunistic_light(self, observation: ThreeBranchesObservation):
        """An unlit lantern/hearth right here to toggle on the way, regardless of role."""
        usable = props.usable(observation)
        if usable is None or usable["type"] not in _LIGHT_TYPES:
            return None
        if usable["id"] in self.used_props:
            return None
        if not _prop_untouched(observation, usable):
            return None
        return usable

    def _advance_hop(self, observation: ThreeBranchesObservation, heading: float, here, target_pos, sweep_at_target=False):
        """Walk toward target_pos in short hops, sweeping at each waypoint before the next leg."""
        if self.hop_point is None:
            self.hop_point = self._pick_hop_point(observation, here, target_pos)
            self._path = None
            self.stuck_ticks = 0

        # On the final leg the caller decides arrival (prop reach is tighter than 2.0), so don't
        # sweep-stop short of the target -- that looped forever just outside use range.
        final_leg = geometry.distance(self.hop_point, target_pos) < 1.0
        if (sweep_at_target or not final_leg) and geometry.distance(here, self.hop_point) < 2.0:
            self.hop_point = None
            return action.stand(heading, "sweep")

        if me.moved(observation) < 0.05:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0
        if self.stuck_ticks >= _STUCK_TICKS_LIMIT:
            self.hop_point = None
            self.stuck_ticks = 0
            self._path = None
            return action.stand(heading, "shrug")

        walk_heading = self._walk_toward(self.hop_point, observation)
        if self._walk_failed:
            if self.role_target is not None and self.role_target["prop_id"] is not None:
                self.used_props.add(self.role_target["prop_id"])
            self.role_target = None
            self.hop_point = None
            self._path = None
            self.stuck_ticks = 0
            self._walk_failed = False
            return action.stand(heading, "shrug")
        return action.walk(walk_heading, 1.0, "none")

    def _idle_hop(self, observation: ThreeBranchesObservation, heading: float, here):
        """Keep moving in short hops when a role has run out of things to do for the day."""
        if self.hop_point is None:
            self.hop_point = self._random_nearby_point(observation, here)
            self._path = None
            self.stuck_ticks = 0
            return action.stand(heading, self.rng.choice(_IDLE_EMOTES))
        return self._advance_hop(observation, heading, here, self.hop_point, sweep_at_target=True)

    def _advance_building(self, observation: ThreeBranchesObservation, heading: float, here):
        """The building-wanderer role: head to a random building's doorway, sweep, pick another."""
        if self.role_target is None:
            candidates = list(layout.buildings(observation))
            self.rng.shuffle(candidates)
            doorway = None
            for building in candidates:
                doorway = layout.doorway(observation, building["id"])
                if doorway is not None:
                    break
            if doorway is None:
                return self._idle_hop(observation, heading, here)
            self.role_target = {"prop_id": None, "type": None, "position": doorway}
            self.hop_point = None

        target_pos = self.role_target["position"]
        if geometry.distance(here, target_pos) <= 2.0:
            self.role_target = None
            self.hop_point = None
            return action.stand(heading, "sweep")

        return self._advance_hop(observation, heading, here, target_pos)

    def _arrival_check(self, observation: ThreeBranchesObservation, heading: float, here):
        """Confirm the role target still needs doing before using it; None means go ahead."""
        target = self.role_target
        start = _PROP_START_STATE.get(target["type"])
        state = props.state(observation, target["prop_id"])
        if start is None or state == start:
            self.arrival_looks = 0
            return None
        if state is None and self.arrival_looks < _ARRIVAL_LOOK_TICKS:
            self.arrival_looks += 1
            prop = props.find(observation, target["prop_id"])
            return action.stand(geometry.heading_to(here, _prop_centre(prop)), "none")
        if state is None:
            return self._give_up_target(observation, heading)  # never came into view
        self._clear_target()  # already done by someone else; memory now knows
        return action.stand(heading, "shrug")
 
    def _clear_target(self) -> None:
        self.role_target = None
        self.hop_point = None
        self.arrival_looks = 0
        self.reposition_tries = 0
        self.claim_told = set()
 
    def _give_up_target(self, observation: ThreeBranchesObservation, heading: float):
        """Drop the role target for now: timed props come back once they'd revert, others don't."""
        target = self.role_target
        if target["type"] in _REVERT_TICKS:
            self.timed_done[target["prop_id"]] = day.tick(observation)
        else:
            self.used_props.add(target["prop_id"])
        self._clear_target()
        return action.stand(heading, "shrug")

    def _arrived(self, here) -> bool:
        reach = _ARRIVAL_SLACK if self.role_target.get("tight") else geometry.PROP_REACH
        return geometry.distance(here, self.role_target["position"]) <= reach

    def _reposition(self, observation: ThreeBranchesObservation, heading: float, here):
        """"use" would pick a different prop: step 90 degrees around the target until it's ours."""
        prop = props.find(observation, self.role_target["prop_id"])
        centre = _prop_centre(prop)
        width, height = _PROP_SIZES.get(prop["type"], (1, 1))
        radius = max(width, height) / 2 + 0.7
        bearing = self.role_target.setdefault("bearing", geometry.heading_to(centre, here))
        while self.reposition_tries < len(_REPOSITION_TURNS):
            angle = math.radians(bearing + _REPOSITION_TURNS[self.reposition_tries])
            self.reposition_tries += 1
            spot = {"x": centre["x"] + radius * math.cos(angle), "y": centre["y"] + radius * math.sin(angle)}
            cell = layout.cell_at(observation, spot)
            if cell is not None and layout.walkable(observation, cell):
                self.role_target["position"] = spot
                self.role_target["tight"] = True
                self.hop_point = None
                return action.stand(geometry.heading_to(here, centre), "none")
        return self._give_up_target(observation, heading)

    def _interested(self, player_id: str, prop_type: str) -> bool:
        """Whether this villager's role works this kind of prop."""
        return people.is_npc(player_id) and prop_type in _ROLE_TARGET_TYPES.get(_role_of(player_id), ())

    def _someone_else_has(self, observation: ThreeBranchesObservation, prop, here, tick: int) -> bool:
        """Is another villager using, claiming, or clearly walking to this prop ahead of us?

        Ties go to whoever is closer to the prop, then the lower player number, so two villagers
        that can see each other reach the same answer without talking.
        """
        prop_id = prop["id"]
        centre = _prop_centre(prop)
        mine = (geometry.distance(here, centre), self.number)
        seen = people.seen(observation)
        for person in seen:
            if person["expression"]["type"] == "use" and person["expression"]["target"] == prop_id:
                return True
        claim = self.claims.get(prop_id)
        if claim is not None and claim[0] != self.player_id and tick - claim[1] <= _CLAIM_TTL_TICKS:
            claimer = people.find(observation, claim[0])
            if claimer is None:
                return True  # out of earshot now; assume they're on it
            if (geometry.distance(claimer["position"], centre), _player_number(claim[0])) < mine:
                return True
        for person in seen:
            if not self._interested(person["id"], prop["type"]) or person["moved"] < 0.05:
                continue
            gap = abs((float(person["heading"]) - geometry.heading_to(person["position"], centre) + 180.0) % 360.0 - 180.0)
            if gap > _HEADING_MATCH_DEGREES:
                continue
            if (geometry.distance(person["position"], centre), _player_number(person["id"])) < mine:
                return True
        return False

    def _share_claim(self, observation: ThreeBranchesObservation) -> None:
        """Tell each villager in earshot who could want our target, once, that we're on it."""
        target = self.role_target
        for person in people.nearby(observation):
            player_id = person["id"]
            if player_id in self.claim_told or not self._interested(player_id, target["type"]):
                continue
            self.claim_told.add(player_id)
            self._outbox.append({"to": player_id, "text": _CLAIM_TEXT.format(target["prop_id"])})

    def _mid_toggle(self) -> bool:
        """Whether a toggle hold is under way; interrupting it would flip the prop back."""
        return self.active_task is not None and self.active_task["type"] in _TOGGLE_TYPES

    def _bell_destination(self, observation: ThreeBranchesObservation) -> dict[str, float] | None:
        """If the bell is ringing and within range, head there; otherwise None."""
        if not day.bell_ringing(observation):
            return None
        here = me.position(observation)
        bell = next((prop for prop in props.all(observation) if prop["type"] == "bell"), None)
        if bell is None:
            return None
        bell_pos = _cell_centre(bell["cell"])
        if geometry.distance(here, bell_pos) > _BELL_RANGE:
            return None
        return bell_pos

    def _cooled(self, player_id: str, tick: int) -> bool:
        """Whether the pair cooldown with this person has run out (or we've never interacted)."""
        entry = self.interactions.get(player_id)
        return entry is None or tick - entry["last"] >= _INTERACTION_COOLDOWN_TICKS

    def _record_interaction(self, player_id: str, tick: int, initiated_by: str) -> None:
        """Log a greeting or reply; a new exchange only counts once the cooldown had run out."""
        entry = self.interactions.get(player_id)
        if entry is None:
            entry = self.interactions[player_id] = {"last": tick, "count": 0, "initiated_by": initiated_by}
        if tick - entry["last"] >= _INTERACTION_COOLDOWN_TICKS or entry["count"] == 0:
            entry["count"] += 1
            entry["initiated_by"] = initiated_by
        entry["last"] = tick
        if people.is_visitor(player_id):
            self.met_visitor_tick = tick
            self.news_told = set()

    def _greeting_text(self, kind: str, player_id: str, tick: int) -> str:
        """The greeting line, plus visitor news for anyone not yet told about a recent meeting."""
        text = _GREETING_LINES[kind]
        recent = self.met_visitor_tick is not None and tick - self.met_visitor_tick <= _VISITOR_NEWS_TICKS
        if kind != "visitor" and recent and player_id not in self.news_told:
            self.news_told.add(player_id)
            text += _VISITOR_NEWS
        return text

    def _greeting_me(self, person, here) -> bool:
        """A wave or nod from someone close by who is facing us."""
        if person["expression"]["type"] not in ("wave", "nod"):
            return False
        if geometry.distance(person["position"], here) > _GREETING_NOTICE_RANGE:
            return False
        toward_me = geometry.heading_to(person["position"], here)
        gap = abs((float(person["heading"]) - toward_me + 180.0) % 360.0 - 180.0)
        return gap <= _GREETING_FACING_DEGREES

    def relation(self, player_id: str) -> str:
        """visitor, family (same home), friend (same role, once seen), acquaintance (has greeted or
        been greeted), else stranger."""
        if people.is_visitor(player_id):
            return "visitor"
        if self.home != "none" and self.homes.get(player_id) == self.home:
            return "family"
        if player_id in self.seen_ever and people.is_npc(player_id) and _role_of(player_id) == self.role:
            return "friend"
        entry = self.interactions.get(player_id)
        if entry is not None and entry["count"] > 0:
            return "acquaintance"
        return "stranger"

    def _may_greet(self, player_id: str, tick: int) -> bool:
        if self.relation(player_id) == "family" and tick < _FAMILY_GREET_AFTER_TICK:
            return False
        return self._cooled(player_id, tick)

    def _greeted_by(self, player_id: str, tick: int) -> None:
        """They greeted us first: turn our own pending greeting into the answer, or owe them one."""
        if self.pending_greet is not None and self.pending_greet["id"] == player_id:
            self.pending_greet["reply"] = True
            self.pending_greet["ready_tick"] = min(self.pending_greet["ready_tick"], tick + _REPLY_DELAY_TICKS)
        elif self._cooled(player_id, tick):
            self.reply_due.setdefault(player_id, tick)

    def _goes_second(self, person, here) -> bool:
        """When two villagers face each other, the lower player number speaks first."""
        if not people.is_npc(person["id"]) or "heading" not in person:
            return False
        toward_me = geometry.heading_to(person["position"], here)
        gap = abs((float(person["heading"]) - toward_me + 180.0) % 360.0 - 180.0)
        return gap <= geometry.VISION_DEGREES / 2 and _player_number(person["id"]) < self.number

    def _say(self, player_id: str, text: str, nearby_ids) -> None:
        """Speak to someone directly, only if they're in earshot (otherwise it would be dropped)."""
        if player_id in nearby_ids:
            self._outbox.append({"to": player_id, "text": text})

    def _react_to_people(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction | None:
        """Stop and turn toward whoever just showed up, or None when nobody is due a reaction.

        Reactions outrank every task, so this runs before the phase logic. Pausing mid-use costs
        nothing: the use counter simply doesn't advance this tick and resumes on the next one.
        """
        here = me.position(observation)
        tick = day.tick(observation)
        seen = people.seen(observation)
        nearby = people.nearby(observation)
        seen_ids = {person["id"] for person in seen}
        # Chat is only delivered to someone in hearing range, so speaking needs this, not sight
        nearby_ids = {person["id"] for person in nearby}
        self.seen_ever.update(seen_ids)

        def find(player_id: str):
            """That person's current record, from sight or hearing, or None once they are gone."""
            for person in seen:
                if person["id"] == player_id:
                    return person
            for person in nearby:
                if person["id"] == player_id:
                    return person
            return None

        # Someone waving or nodding at us: inside the cooldown it's their reply (it still counts and
        # restarts the cooldown); outside it, we owe them an answer
        for person in seen:
            if not self._greeting_me(person, here):
                continue
            if self._cooled(person["id"], tick):
                self._greeted_by(person["id"], tick)
            else:
                self._record_interaction(person["id"], tick, "them")

        # Already noticed someone: walk to them if they started close by, then greet
        if self.pending_greet is not None:
            person = find(self.pending_greet["id"])
            if person is None:
                self.pending_greet = None  # they left before we could say anything
            else:
                facing = geometry.heading_to(here, person["position"])
                dist_to_person = geometry.distance(here, person["position"])

                # If the visitor started within approach range, walk over before greeting
                if self.pending_greet.get("approach_player") and dist_to_person > _VISITOR_GREET_DISTANCE:
                    return action.walk(self._walk_toward(person["position"], observation), 1.0, "none")

                if tick < self.pending_greet["ready_tick"]:
                    return action.stand(facing, "none")

                player_id = person["id"]
                reply = self.pending_greet.get("reply")
                kind = self.relation(player_id)  # read before recording, so a first meeting is still "stranger"
                if player_id not in nearby_ids:
                    # Too far to be heard: a silent wave or nod, and no exchange is logged, so a real
                    # one still happens when they come into earshot
                    self.pending_greet = None
                    self.distant_waves[player_id] = tick
                    return action.stand(facing, "wave" if kind in _WAVE_RELATIONS else "nod")
                self._record_interaction(player_id, tick, "them" if reply else "me")
                self.pending_greet = None
                text = _REPLY_LINES[kind] if reply else self._greeting_text(kind, player_id, tick)
                self._say(player_id, text, nearby_ids)
                return action.stand(facing, "wave" if kind in _WAVE_RELATIONS else "nod")

        def greet_due(player_id: str) -> bool:
            if tick - self.distant_waves.get(player_id, -_DISTANT_WAVE_COOLDOWN_TICKS) < _DISTANT_WAVE_COOLDOWN_TICKS:
                return False
            return self._may_greet(player_id, tick)

        def greet_friend_in_passing(player_id: str, reply: bool) -> None:
            """Friends don't stop: say the line and wave while carrying on with whatever we're doing."""
            self._record_interaction(player_id, tick, "them" if reply else "me")
            text = _REPLY_LINES["friend"] if reply else self._greeting_text("friend", player_id, tick)
            self._say(player_id, text, nearby_ids)
            self._passing_emote = "wave"

        def startle_due(player_id: str) -> bool:
            return tick - self.startled.get(player_id, -_STARTLE_COOLDOWN_TICKS) >= _STARTLE_COOLDOWN_TICKS

        def notice(person, expression: str, reply: bool = False):
            """Turn toward someone and start the beat that ends in a wave or a nod."""
            approach = (
                not reply
                and people.is_visitor(person["id"])
                and geometry.distance(here, person["position"]) <= _VISITOR_APPROACH_RANGE
            )
            delay = _REPLY_DELAY_TICKS if reply else _GREET_DELAY_TICKS
            if not reply and self._goes_second(person, here):
                delay += _GREET_DELAY_TICKS  # let them speak first; we greet anyway if they don't
            self.pending_greet = {
                "id": person["id"],
                "ready_tick": tick + delay,
                "approach_player": approach,
                "reply": reply,
            }
            return action.stand(geometry.heading_to(here, person["position"]), expression)

        # Answer anyone who greeted us first, while they're still around
        for player_id, since in list(self.reply_due.items()):
            del self.reply_due[player_id]
            person = find(player_id)
            if person is None or tick - since > _REPLY_WINDOW_TICKS or not greet_due(player_id):
                continue
            if self.relation(player_id) == "friend":
                if player_id in nearby_ids:
                    greet_friend_in_passing(player_id, reply=True)
                    return None
                continue
            return notice(person, "none", reply=True)

        # The visitor comes first
        visitor = next((person for person in seen if people.is_visitor(person["id"])), None)
        if visitor is not None and greet_due(visitor["id"]):
            return notice(visitor, "none")

        # Friends in earshot get a greeting on the move; friends farther off wait until they're closer
        for person in nearby:
            if self.relation(person["id"]) != "friend" or not greet_due(person["id"]):
                continue
            sighted = find(person["id"]) if person["id"] in seen_ids else person
            if sighted is not person and self._goes_second(sighted, here):
                continue  # they'll greet us first and we answer in passing
            greet_friend_in_passing(person["id"], reply=False)
            return None

        for person in seen:
            player_id = person["id"]
            if people.is_visitor(player_id) or self.relation(player_id) == "friend":
                continue
            if greet_due(player_id):
                return notice(person, "none")

        # Heard but not seen means they are in the blind spot behind us, so the beat opens with a
        # startle instead. It doesn't use up the greet cooldown, so the wave or nod still follows.
        for person in nearby:
            if person["id"] in seen_ids:
                continue
            if self.relation(person["id"]) == "friend":
                continue
            if startle_due(person["id"]) and greet_due(person["id"]):
                self.startled[person["id"]] = tick
                return notice(person, "startle")

        return None

    def _family(self) -> list[str]:
        return [pid for pid, home in self.homes.items() if home == self.home and pid != self.player_id]

    def _family_time_due(self, tick: int) -> bool:
        return (
            not self.family_lunch_done
            and self.home_doorway is not None
            and _MIDDAY_TICKS[0] <= tick <= _MIDDAY_TICKS[1]
            and bool(self._family())
        )

    def _family_lunch_tick(self, observation: ThreeBranchesObservation, heading: float, here, tick: int):
        """Walk to the home doorway, wait for family, spend a little time together, then go back to work."""
        lunch = self.family_lunch

        def finish():
            self.family_lunch = None
            self.family_lunch_done = True
            self._path = None
            return action.stand(heading, "wave")

        if tick > _MIDDAY_TICKS[1] and lunch["stage"] != "together":
            return finish()

        if lunch["stage"] == "going":
            # Family already standing at the door (and blocking it) counts as having arrived
            near_family = any(
                person is not None and geometry.distance(here, person["position"]) <= _FAMILY_MEET_DISTANCE
                for person in (people.find(observation, player_id) for player_id in self._family())
            )
            if not near_family and geometry.distance(here, self.home_doorway) > 1.5:
                if tick - lunch["since"] > _FAMILY_TRAVEL_TICKS:
                    return finish()
                walk_heading = self._walk_toward(self.home_doorway, observation)
                if self._walk_failed:
                    return finish()
                return action.walk(walk_heading, 1.0, "none")
            lunch["stage"], lunch["since"] = "waiting", tick

        if lunch["stage"] == "waiting":
            for player_id in self._family():
                person = people.find(observation, player_id)
                if person is not None and geometry.distance(here, person["position"]) <= _FAMILY_MEET_DISTANCE:
                    lunch.update(stage="together", since=tick, partner=player_id)
                    return action.stand(geometry.heading_to(here, person["position"]), "wave")
            if tick - lunch["since"] > _FAMILY_WAIT_TICKS:
                return finish()
            return action.stand(heading, "none")

        # Together: face each other, with the odd laugh or nod
        if tick - lunch["since"] >= _FAMILY_TIME_TICKS:
            return finish()
        partner = people.find(observation, lunch["partner"])
        facing = geometry.heading_to(here, partner["position"]) if partner is not None else heading
        expression = self.rng.choice(("laugh", "nod")) if (tick - lunch["since"]) % 12 == 6 else "none"
        return action.stand(facing, expression)

    def _task_tick(self, observation: ThreeBranchesObservation, heading: float, here) -> ThreeBranchesAction:
        """One tick of role-driven work: answer the bell, grab a passing light, else do the job."""

        # The bell outranks the day's role work, and is answered directly (no hops -- be quick)
        bell_destination = None if self._mid_toggle() else self._bell_destination(observation)
        if bell_destination is not None:
            if self.current_destination != bell_destination:
                self.current_destination = bell_destination
                self._path = None
                self.stuck_ticks = 0
            if geometry.distance(here, self.current_destination) < 2.0:
                self.current_destination = None
                return action.stand(heading, "sweep")
            return action.walk(self._walk_toward(self.current_destination, observation), 1.0, "none")

        # Everyone, whatever their role, lights any unlit lantern/hearth they pass close enough to use
        if self.active_task is None:
            light = self._opportunistic_light(observation)
            if light is not None:
                self.active_task = {
                    "prop_id": light["id"],
                    "type": light["type"],
                    "ticks_remaining": self._prop_use_ticks(light["type"]),
                }
                self.use_counter = 0
            elif self.role_target is not None and self.role_target["prop_id"] is not None:
                if self._arrived(here):
                    check = self._arrival_check(observation, heading, here)
                    if check is not None:
                        return check
                    usable = props.usable(observation)
                    if usable is None or usable["id"] != self.role_target["prop_id"]:
                        return self._reposition(observation, heading, here)
                    self.active_task = {
                        "prop_id": self.role_target["prop_id"],
                        "type": self.role_target["type"],
                        "ticks_remaining": self._prop_use_ticks(self.role_target["type"]),
                    }
                    self.use_counter = 0

        if self.active_task is not None:
            if self.use_counter >= self.active_task["ticks_remaining"]:
                if self.active_task["type"] in _REVERT_TICKS:
                    self.timed_done[self.active_task["prop_id"]] = day.tick(observation)  # back in the pool once it reverts
                else:
                    self.used_props.add(self.active_task["prop_id"])
                if self.role_target is not None and self.role_target["prop_id"] == self.active_task["prop_id"]:
                    self.role_target = None
                    self.hop_point = None
                    # Start a break after completing a task
                    self.break_ticks_remaining = self.rng.randint(_BREAK_MIN_TICKS, _BREAK_MAX_TICKS)
                self.active_task = None
                self.use_counter = 0
            else:
                self.use_counter += 1
                return action.stand(heading, "use")

        # First break (or gap between targets) at midday: go home and spend time with family
        tick = day.tick(observation)
        if (
            self.family_lunch is None
            and self.active_task is None
            and (self.break_ticks_remaining > 0 or self.role_target is None)
            and self._family_time_due(tick)
        ):
            self.family_lunch = {"stage": "going", "since": tick}
            self.break_ticks_remaining = 0
            self.hop_point = None
            self._path = None
        if self.family_lunch is not None:
            return self._family_lunch_tick(observation, heading, here, tick)

        # Break after completing a task: idle-hop for a random duration (still lighting lamps en route)
        if self.break_ticks_remaining > 0:
            self.break_ticks_remaining -= 1
            return self._idle_hop(observation, heading, here)

        if self.role == "building_wanderer":
            return self._advance_building(observation, heading, here)

        if self.role_target is None:
            prop = self._random_available_prop(observation, _ROLE_TARGET_TYPES[self.role])
            if prop is None:
                return self._idle_hop(observation, heading, here)  # nothing left today; keep it lively
            self.role_target = {
                "prop_id": prop["id"],
                "type": prop["type"],
                "position": _cell_centre(_nearest_walkable_cell(observation, prop["cell"])),
            }
            self.hop_point = None
            self.reposition_tries = 0
            self.claim_told = set()

        target_prop = props.find(observation, self.role_target["prop_id"])
        if self._someone_else_has(observation, target_prop, here, day.tick(observation)):
            self._clear_target()  # someone else has it; pick the next one next tick
            return action.stand(heading, "nod")
        self._share_claim(observation)

        if self._arrived(here):
            return action.stand(heading, "none")  # arrived; active_task starts from here next tick

        return self._advance_hop(observation, heading, here, self.role_target["position"])

    def act(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        """Choose this tick's action, adding a wave if we greeted a friend on the move."""
        self._passing_emote = None
        chosen = self._decide(observation)
        if self._passing_emote is not None and int(chosen["action"]) == 0:
            return action.walk(float(chosen["heading"]), float(chosen["speed"]), self._passing_emote)
        return chosen

    def _decide(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        """Choose one action from current observation and persistent day state."""

        here = me.position(observation)
        heading = me.heading(observation)
        tick = day.tick(observation)
        for seen_prop in props.seen(observation):
            self.prop_memory[seen_prop["prop"]] = (seen_prop["state"], tick)

        # Announce role once at start of day
        if not self.announced_role:
            self.announced_role = True
            self._outbox.append({"to": None, "text": f"I'm a {self.role.replace('_', ' ')}."})
            return action.stand(heading, "none")

        mid_toggle = self._mid_toggle()
        if not mid_toggle:
            reaction = self._react_to_people(observation)
            if reaction is not None:
                return reaction

        if self.phase != "returning" and tick >= _RETURN_HOME_TICK and not mid_toggle:
            # Time to head home, whatever else was in progress
            self.phase = "returning"
            self.active_task = None
            self.role_target = None
            self.hop_point = None
            self.current_destination = None
            self._path = None
            self.stuck_ticks = 0

        if self.phase == "task":
            return self._task_tick(observation, heading, here)

        # Phase: returning (walk home for the night, then sleep for the rest of the day)
        if self.phase == "returning":
            if self.home_doorway is None:  # no home to return to (e.g. the visitor seat)
                return action.stand(heading, "sleep")
            here_cell = layout.cell_at(observation, here)
            ground = layout.ground_at(observation, here_cell) if here_cell is not None else None
            if ground == "interior" or geometry.distance(here, self.home_doorway) < 1.0:
                return action.stand(heading, "sleep")
            return action.walk(self._walk_toward(self.home_doorway, observation), 1.0, "none")

        return action.stand(heading, "none")

    # Optional: messaging. On your turn, chat receives messages addressed to your player since
    # its previous turn. Return messages with a recipient and text, or nothing to stay silent.
    # Use None as the recipient to broadcast. Every message is recorded and shown in replays.
    #
    def chat(self, inbox: list[dict]) -> list[dict] | None:
        """Note task claims from other villagers, then send every message act() queued."""
        for message in inbox or ():
            sender = message.get("from")
            if not isinstance(sender, str) or not people.is_npc(sender):
                continue
            match = _CLAIM_PATTERN.fullmatch(str(message.get("text", "")))
            if match:
                self.claims[match.group(1)] = (sender, int(message.get("tick", 0)))
        for message in inbox or ():
            sender = message.get("from")
            if not isinstance(sender, str) or message.get("to") != self.player_id:
                continue
            if _CLAIM_PATTERN.fullmatch(str(message.get("text", ""))):
                continue
            sent = int(message.get("tick", 0))
            if self._cooled(sender, sent):
                self._greeted_by(sender, sent)  # spoken to directly: answer them
        if not self._outbox:
            return None
        messages, self._outbox = self._outbox, []
        return messages
