"""Self-contained Streamlit pathfinding and network-routing simulator.

Run with: streamlit run app.py
This file deliberately contains its algorithms, models, renderers, and UI so
Streamlit Cloud does not depend on local project-package imports.
"""
from collections import deque
from heapq import heappop, heappush
from math import sqrt
from random import Random
from time import perf_counter, sleep

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st


# ---------------------------------------------------------------------------
# Grid model and search algorithms
# ---------------------------------------------------------------------------
def make_grid(size: int, density: int, seed: int) -> set[tuple[int, int]]:
    rng = Random(seed)
    walls = {(r, c) for r in range(size) for c in range(size)
             if rng.random() < density / 100}
    walls.discard((0, 0))
    walls.discard((size - 1, size - 1))
    return walls


def grid_neighbors(point, size, walls, diagonal=False):
    r, c = point
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    found = []
    for dr, dc in offsets:
        nxt = (r + dr, c + dc)
        if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or nxt in walls:
            continue
        if dr and dc and ((r + dr, c) in walls or (r, c + dc) in walls):
            continue
        found.append(nxt)
    return found


def grid_search(size, walls, start, goal, algorithm="BFS", diagonal=False):
    """Return path, visit order, max frontier size, and elapsed milliseconds."""
    began = perf_counter()
    parents = {start: None}
    distances = {start: 0.0}
    visited = set()
    visit_order = []
    max_frontier = 1
    serial = 0
    if algorithm == "BFS":
        frontier = deque([start])
    else:
        frontier = [(0.0, serial, start)]

    def heuristic(point):
        dr, dc = abs(point[0] - goal[0]), abs(point[1] - goal[1])
        if diagonal:
            return max(dr, dc) + (sqrt(2) - 1) * min(dr, dc)
        return dr + dc

    while frontier:
        if algorithm == "BFS":
            current = frontier.popleft()
        else:
            _, _, current = heappop(frontier)
            if current in visited:
                continue
        visited.add(current)
        visit_order.append(current)
        if current == goal:
            break

        for nxt in grid_neighbors(current, size, walls, diagonal):
            step_cost = sqrt(2) if nxt[0] != current[0] and nxt[1] != current[1] else 1.0
            tentative = distances[current] + step_cost
            if algorithm == "BFS":
                if nxt in parents:
                    continue
                parents[nxt] = current
                distances[nxt] = distances[current] + 1
                frontier.append(nxt)
            elif tentative < distances.get(nxt, float("inf")):
                parents[nxt] = current
                distances[nxt] = tentative
                serial += 1
                if algorithm == "A*":
                    score = tentative + heuristic(nxt)
                elif algorithm == "Greedy":
                    score = heuristic(nxt)
                else:  # Dijkstra
                    score = tentative
                heappush(frontier, (score, serial, nxt))
        max_frontier = max(max_frontier, len(frontier))

    path = []
    if goal in parents:
        point = goal
        while point is not None:
            path.append(point)
            point = parents[point]
        path.reverse()
    return {
        "path": path,
        "visited": visit_order,
        "visited_count": len(visit_order),
        "max_frontier": max_frontier,
        "elapsed_ms": (perf_counter() - began) * 1000,
    }


def bfs_search_frames(size, walls, start, goal, diagonal=False):
    """Return faithful BFS queue snapshots for the Streamlit walkthrough."""
    queue = deque([start])
    parents = {start: None}
    visited = []
    frames = []
    max_queue = 1
    while queue:
        current = queue.popleft()
        visited.append(current)
        if current == goal:
            path = []
            point = goal
            while point is not None:
                path.append(point)
                point = parents[point]
            path.reverse()
            frames.append({"current": current, "visited": list(visited),
                           "frontier": list(queue), "max_queue": max_queue,
                           "path": path})
            return frames
        for neighbor in grid_neighbors(current, size, walls, diagonal):
            if neighbor not in parents:
                parents[neighbor] = current
                queue.append(neighbor)
        max_queue = max(max_queue, len(queue))
        frames.append({"current": current, "visited": list(visited),
                       "frontier": list(queue), "max_queue": max_queue,
                       "path": []})
    return frames


def draw_grid(size, walls, result=None, start=(0, 0), goal=None,
              visited=None, frontier=None, current=None):
    goal = goal or (size - 1, size - 1)
    # 0 free, 1 wall, 2 visited, 3 route, 4 source, 5 goal, 6 frontier, 7 active
    cells = [[0] * size for _ in range(size)]
    for r, c in walls:
        cells[r][c] = 1
    active_visited = visited if visited is not None else (result or {}).get("visited", [])
    active_path = (result or {}).get("path", [])
    for r, c in active_visited:
        if cells[r][c] == 0:
            cells[r][c] = 2
    for r, c in (frontier or []):
        if cells[r][c] == 0:
            cells[r][c] = 6
    for r, c in active_path:
        if cells[r][c] in (0, 2, 6):
            cells[r][c] = 3
    if current is not None and current not in (start, goal):
        cells[current[0]][current[1]] = 7
    cells[start[0]][start[1]] = 4
    cells[goal[0]][goal[1]] = 5
    fig, ax = plt.subplots(figsize=(min(11, size * .32), min(8, size * .32)))
    ax.imshow(cells, cmap=ListedColormap([
        "#f1f5f9", "#202938", "#83b6e8", "#fbbf24", "#10b981",
        "#f43f5e", "#bfdbfe", "#00bceb"
    ]), vmin=0, vmax=7)
    ax.set_xticks([x - .5 for x in range(1, size)], minor=True)
    ax.set_yticks([y - .5 for y in range(1, size)], minor=True)
    ax.grid(which="minor", color="#cbd5e1", linewidth=.45)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    fig.tight_layout(pad=.1)
    return fig


# ---------------------------------------------------------------------------
# Cisco-style weighted topology
# ---------------------------------------------------------------------------
NETWORK = {
    "START": {"A": 2.0, "E": 6.0, "D": 8.0},
    "A": {"B": 3.0, "C": 6.75, "E": 5.0},
    "B": {"C": 5.0, "E": 2.67},
    "C": {"TARGET": 6.0, "G": 2.75},
    "E": {"TARGET": 3.0, "D": 1.0},
    "D": {"TARGET": 4.25, "G": 5.0},
    "TARGET": {"G": 9.0},
    "G": {},
}
NODE_POS = {
    "START": (.08, .48), "A": (.28, .78), "B": (.48, .78),
    "C": (.68, .78), "E": (.40, .30), "D": (.60, .30),
    "TARGET": (.84, .50), "G": (.94, .22),
}


def get_active_network(overrides):
    """Apply user-selected failures and congestion to the base topology."""
    active = {}
    for node, neighbors in NETWORK.items():
        active[node] = {}
        for neighbor, weight in neighbors.items():
            state = overrides.get(f"{node} → {neighbor}", "Normal")
            if state == "Failed":
                continue
            active[node][neighbor] = weight * (5 if state == "Congested (5x)" else 1)
    return active


def network_dijkstra(source, target, active_net):
    distances = {node: float("inf") for node in active_net}
    parents = {node: None for node in active_net}
    distances[source] = 0.0
    queue = [(0.0, source)]
    visited = []
    trace = []
    while queue:
        cost, node = heappop(queue)
        if node in visited or cost != distances.get(node, float("inf")):
            continue
        visited.append(node)
        trace.append({"Step": len(trace) + 1, "Event": "Process router",
                      "Router": node, "Cost": f"{cost:.2f}"})
        if node == target:
            break
        for neighbor, weight in active_net.get(node, {}).items():
            candidate = cost + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                parents[neighbor] = node
                heappush(queue, (candidate, neighbor))
                trace.append({"Step": len(trace) + 1, "Event": "Relax link",
                              "Router": f"{node} → {neighbor}",
                              "Cost": f"{weight:.2f} (total {candidate:.2f})"})
    path = []
    if distances[target] < float("inf"):
        node = target
        while node is not None:
            path.append(node)
            node = parents[node]
        path.reverse()
    table = [{"Router": node,
              "Distance": "∞" if distances.get(node, float("inf")) == float("inf") else f"{distances[node]:.2f}",
              "Next hop": parents.get(node) or "—",
              "Status": "Visited" if node in visited else "Unvisited"}
             for node in active_net]
    return {"source": source, "target": target, "distances": distances,
            "parents": parents, "visited": visited, "path": path,
            "cost": distances.get(target, float("inf")), "table": table, "trace": trace}


def network_bellman_ford(source, target, active_net):
    distances = {node: float("inf") for node in active_net}
    parents = {node: None for node in active_net}
    distances[source] = 0.0
    trace = []
    for _ in range(max(0, len(active_net) - 1)):
        changed = False
        for node, neighbors in active_net.items():
            if distances[node] == float("inf"):
                continue
            for neighbor, weight in neighbors.items():
                candidate = distances[node] + weight
                if candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    parents[neighbor] = node
                    changed = True
                    trace.append({"Step": len(trace) + 1, "Event": "Relax link",
                                  "Router": f"{node} → {neighbor}",
                                  "Cost": f"{weight:.2f} (total {candidate:.2f})"})
        if not changed:
            break
    path = []
    if distances[target] < float("inf"):
        node = target
        while node is not None:
            path.append(node)
            node = parents[node]
        path.reverse()
    reached = {node for node, cost in distances.items() if cost < float("inf")}
    table = [{"Router": node,
              "Distance": "∞" if distances[node] == float("inf") else f"{distances[node]:.2f}",
              "Next hop": parents[node] or "—",
              "Status": "Reachable" if node in reached else "Unreachable"}
             for node in active_net]
    return {"source": source, "target": target, "distances": distances,
            "parents": parents, "visited": list(reached), "path": path,
            "cost": distances[target], "table": table, "trace": trace}


def draw_network(result, active_net):
    fig, ax = plt.subplots(figsize=(10, 5.8), facecolor="#080e18")
    ax.set_facecolor("#080e18")
    route_edges = set(zip(result["path"], result["path"][1:]))
    for node, neighbors in NETWORK.items():
        x1, y1 = NODE_POS[node]
        for neighbor, base_weight in neighbors.items():
            x2, y2 = NODE_POS[neighbor]
            hot = (node, neighbor) in route_edges
            is_active = neighbor in active_net.get(node, {})
            weight = active_net.get(node, {}).get(neighbor, base_weight)
            color = "#00bceb" if hot else ("#334155" if is_active else "#64748b")
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops={"arrowstyle": "->", "color": color,
                                    "linestyle": "-" if is_active else ":",
                                    "lw": 2.8 if hot else 1.1, "shrinkA": 18, "shrinkB": 18})
            label = (f"{weight:g}" if weight == base_weight else f"{weight:g} (5x)") if is_active else "FAILED"
            label_color = "#fbbf24" if weight == base_weight else ("#f43f5e" if is_active else "#94a3b8")
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + .025, label,
                    color=label_color, fontsize=8, ha="center",
                    bbox={"facecolor": "#080e18", "edgecolor": "none", "pad": 1})
    for node, (x, y) in NODE_POS.items():
        color = "#10b981" if node == result["source"] else "#f43f5e" if node == result["target"] else "#00bceb"
        ax.scatter([x], [y], s=850, c="#0f172a", edgecolors=color, linewidths=2, zorder=3)
        ax.text(x, y, node, color="white", fontsize=8, fontweight="bold",
                ha="center", va="center", zorder=4)
    ax.set_xlim(0, 1.03)
    ax.set_ylim(.05, .95)
    ax.axis("off")
    fig.tight_layout(pad=.5)
    return fig


# ---------------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Network Algorithms Lab", page_icon="🧭", layout="wide")
st.markdown("""
<style>
.stApp { background:#080e18; color:#e2e8f0; }
[data-testid="stSidebar"] { background:#0f172a; }
.block-container { padding-top:1.1rem; max-width:1500px; }
div[data-testid="stMetric"] { background:#0f172a; border:1px solid #1e293b;
padding:.75rem; border-radius:.7rem; }
</style>
""", unsafe_allow_html=True)
st.title("🧭 Network Algorithms Lab")
st.caption("CS402 · ROUTING PROTOCOLS & PATHFINDING")
grid_tab, network_tab = st.tabs(["▦  Grid Pathfinding", "⌘  Cisco Topology Dijkstra"])

with st.sidebar:
    st.header("Grid simulator")
    size = st.slider("Grid size", 10, 40, 20)
    density = st.slider("Obstacle density", 0, 50, 25, format="%d%%")
    seed = st.number_input("Map seed", 0, 999_999, 7)
    algorithm = st.selectbox("Algorithm", ["A*", "BFS", "Dijkstra", "Greedy"])
    diagonal = st.checkbox("Allow diagonal neighbors", False)
    edit_tool = st.selectbox("Map editing tool", ["Wall", "Source", "Goal"])
    edit_cols = st.columns(2)
    with edit_cols[0]:
        edit_r = st.number_input("Row", 0, size - 1, 0)
    with edit_cols[1]:
        edit_c = st.number_input("Column", 0, size - 1, 0)
    apply_edit = st.button("Apply cell edit", use_container_width=True)
    randomize = st.button("🎲 Randomize obstacles", use_container_width=True)
    clear_walls = st.button("▱ Clear all walls", use_container_width=True)
    run_search = st.button("▶ Find path", type="primary", use_container_width=True)
    run_bfs_demo = st.button("Step through BFS", use_container_width=True)
    animate_delay = st.slider("Walkthrough speed", 0.0, 0.08, 0.02, 0.01)

signature = (size, density, int(seed))
if "signature" not in st.session_state:
    st.session_state.signature = None
if signature != st.session_state.signature or randomize:
    next_seed = int(seed) + (1 if randomize else 0)
    st.session_state.walls = make_grid(size, density, next_seed)
    st.session_state.start = (0, 0)
    st.session_state.goal = (size - 1, size - 1)
    st.session_state.signature = signature
    st.session_state.search_result = None
walls = st.session_state.walls
start, goal = st.session_state.start, st.session_state.goal
if apply_edit:
    point = (int(edit_r), int(edit_c))
    if edit_tool == "Source" and point != goal:
        start = point
        walls.discard(point)
        st.session_state.start = point
    elif edit_tool == "Goal" and point != start:
        goal = point
        walls.discard(point)
        st.session_state.goal = point
    elif edit_tool == "Wall" and point not in (start, goal):
        walls.add(point)
    elif point in (start, goal) and edit_tool != "Wall":
        st.warning("Source and goal must occupy different cells.")
    st.session_state.search_result = None
    start, goal = st.session_state.start, st.session_state.goal
if clear_walls:
    walls.clear()
    st.session_state.search_result = None

if run_search:
    st.session_state.search_result = grid_search(size, walls, start, goal, algorithm, diagonal)

with grid_tab:
    st.subheader("Grid Search Engine")
    st.caption("BFS wavefront and heuristic search through an editable obstacle grid.")
    if run_bfs_demo:
        st.markdown("#### BFS frontier walkthrough")
        placeholder = st.empty()
        info = st.empty()
        frames = bfs_search_frames(size, walls, start, goal, diagonal)
        final_frame = frames[-1] if frames else {}
        for frame in frames:
            placeholder.pyplot(draw_grid(size, walls, None, start, goal,
                                         visited=frame["visited"],
                                         frontier=frame["frontier"],
                                         current=frame["current"]),
                              use_container_width=True)
            info.caption(f"Expanded {len(frame['visited'])} cells · current frontier {len(frame['frontier'])} · peak queue size {frame['max_queue']}")
            if animate_delay:
                sleep(animate_delay)
        if final_frame.get("path"):
            placeholder.pyplot(draw_grid(size, walls,
                                         {"visited": final_frame["visited"], "path": final_frame["path"]},
                                         start, goal), use_container_width=True)
            st.success(f"BFS found a route in {len(final_frame['path']) - 1} moves; peak queue size {final_frame['max_queue']}.")
        else:
            st.warning(f"No route found. BFS peak queue size: {final_frame.get('max_queue', 0)}.")

    visual, summary = st.columns([3, 1])
    with visual:
        st.pyplot(draw_grid(size, walls, st.session_state.search_result, start, goal),
                  use_container_width=True)
    with summary:
        st.markdown("#### Search telemetry")
        result = st.session_state.search_result
        st.markdown("🟩 Source · 🟥 Goal · ⬛ Wall\n\n🟦 Visited · 🟨 Route · 🔹 Frontier")
        if result is None:
            st.info("Choose settings and find a path.")
        elif result["path"]:
            st.success("Route found")
            st.metric("Path length", len(result["path"]) - 1, "moves")
            st.metric("Expanded cells", result["visited_count"])
            st.metric("Search time", f"{result['elapsed_ms']:.2f} ms")
        else:
            st.error("No route exists on this map.")
            st.metric("Expanded cells", result["visited_count"])

    with st.expander("How search algorithms compare"):
        st.markdown("""
- **BFS**: Explores layer by layer and finds the fewest moves on an unweighted grid.
- **Dijkstra**: Finds a minimum-cost route using movement costs.
- **A***: Uses path cost plus a distance heuristic to guide the search toward the goal.
- **Greedy Best-First**: Uses only the heuristic. It can be faster, but does not guarantee the lowest-cost route.

BFS uses a first-in, first-out queue. Its queue has no fixed capacity; the walkthrough reports the peak reached on the current map. Runtime and memory are **O(V + E)**.
        """)

    with st.expander("Benchmark BFS, Dijkstra, and A*"):
        if st.button("Run benchmark on current map", key="run_benchmark"):
            benchmark_rows = []
            for name in ("BFS", "Dijkstra", "A*", "Greedy"):
                data = grid_search(size, walls, start, goal, name, diagonal)
                benchmark_rows.append({"Algorithm": name, "Found": bool(data["path"]),
                                       "Path moves": len(data["path"]) - 1 if data["path"] else None,
                                       "Expanded cells": data["visited_count"],
                                       "Time (ms)": round(data["elapsed_ms"], 3)})
            st.dataframe(benchmark_rows, hide_index=True, use_container_width=True)
            st.bar_chart({"Algorithm": [row["Algorithm"] for row in benchmark_rows],
                          "Expanded cells": [row["Expanded cells"] for row in benchmark_rows]},
                         x="Algorithm", y="Expanded cells")

with network_tab:
    st.subheader("Cisco-style weighted topology")
    st.caption("Simulate shortest-path routing over weighted router links with live link failures and congestion.")
    nodes = list(NETWORK)
    with st.expander("🛠️ Interactive Link Congestion & Failure Simulator", expanded=True):
        all_links = [f"{router} → {neighbor}"
                     for router, neighbors in NETWORK.items() for neighbor in neighbors]
        failure_col, congestion_col = st.columns(2)
        with failure_col:
            failed_links = st.multiselect(
                "Simulate link failures (infinite cost)", all_links, key="failed_links_ui"
            )
        with congestion_col:
            congested_links = st.multiselect(
                "Simulate congestion (5× weight)", all_links, key="congested_links_ui"
            )
        st.session_state.link_overrides = {
            link: ("Failed" if link in failed_links else
                   "Congested (5x)" if link in congested_links else "Normal")
            for link in all_links
        }

    active_net = get_active_network(st.session_state.link_overrides)
    control_cols = st.columns([1, 1, 1, 1])
    with control_cols[0]:
        source = st.selectbox("Source router", nodes, index=0, key="network_source")
    with control_cols[1]:
        destination = st.selectbox("Destination router", nodes, index=6, key="network_target")
    with control_cols[2]:
        route_algorithm = st.selectbox("Routing protocol", ["Dijkstra (SPF)", "Bellman–Ford"])
    with control_cols[3]:
        st.write("")
        run_spf = st.button("▶ Run route calculation", type="primary", use_container_width=True)
    calc_key = (source, destination, route_algorithm,
                tuple(sorted(st.session_state.link_overrides.items())))
    if run_spf or "network_result" not in st.session_state or st.session_state.get("network_calc_key") != calc_key:
        if route_algorithm == "Dijkstra (SPF)":
            st.session_state.network_result = network_dijkstra(source, destination, active_net)
        else:
            st.session_state.network_result = network_bellman_ford(source, destination, active_net)
        st.session_state.network_calc_key = calc_key
    network_result = st.session_state.network_result
    graph_col, table_col = st.columns([1.8, 1])
    with graph_col:
        st.pyplot(draw_network(network_result, active_net), use_container_width=True)
    with table_col:
        st.markdown("#### Routing table")
        st.dataframe(network_result["table"], hide_index=True, use_container_width=True)
        if network_result["path"]:
            st.success("Shortest route: " + " → ".join(network_result["path"]))
            st.metric("Total link cost", f"{network_result['cost']:.2f}")
        else:
            st.warning("No route is available with the current link conditions.")
        st.markdown("#### SPF processing order")
        st.write(" → ".join(network_result["visited"]))
    with st.expander("Protocol relaxation trace"):
        st.dataframe(network_result["trace"], hide_index=True, use_container_width=True)

st.caption("Network Algorithms Lab · Self-contained Streamlit application")
