import time
import streamlit as st
from algorithms.bfs_generator import bfs_search_generator
from environment.grid import Grid, Point
from visualization.draw_grid import draw_grid

def run_interactive_bfs_player(
    grid: Grid,
    start: Point,
    goal: Point,
    diagonal: bool = False,
    delay: float = 0.03,
):
    # 1. Initialize Session State for Playback
    if "step_idx" not in st.session_state:
        st.session_state.step_idx = 0
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "cached_states" not in st.session_state or st.session_state.get("grid_signature") != (start, goal, diagonal):
        # Precompute generator steps so we can jump back and forth instantly
        st.session_state.cached_states = list(bfs_search_generator(grid, start, goal, diagonal))
        st.session_state.step_idx = 0
        st.session_state.grid_signature = (start, goal, diagonal)

    states = st.session_state.cached_states
    max_steps = len(states) - 1 if states else 0

    # 2. Render Control Buttons
    col1, col2, col3, col4, col5 = st.columns(5)
    
    if col1.button("▶ Play", use_container_width=True):
        st.session_state.playing = True
        st.rerun()
        
    if col2.button("⏸ Pause", use_container_width=True):
        st.session_state.playing = False
        st.rerun()
        
    if col3.button("⏭ Step", use_container_width=True):
        st.session_state.playing = False
        if st.session_state.step_idx < max_steps:
            st.session_state.step_idx += 1
        st.rerun()
        
    if col4.button("⏮ Rewind", use_container_width=True):
        st.session_state.playing = False
        if st.session_state.step_idx > 0:
            st.session_state.step_idx -= 1
        st.rerun()

    if col5.button("🔄 Reset", use_container_width=True):
        st.session_state.playing = False
        st.session_state.step_idx = 0
        st.rerun()

    # Slider for manual scrubbing
    selected_step = st.slider(
        "Animation Progress",
        min_value=0,
        max_value=max_steps,
        value=st.session_state.step_idx,
        key="slider_progress"
    )
    if selected_step != st.session_state.step_idx:
        st.session_state.step_idx = selected_step
        st.session_state.playing = False
        st.rerun()

    # 3. Retrieve Current State & Render View
    current_state = states[st.session_state.step_idx]

    board_container = st.empty()
    status_container = st.empty()
    metrics_container = st.empty()

    board_container.pyplot(
        draw_grid(
            grid,
            None,
            start,
            goal,
            visited=current_state["visited"],
            frontier=current_state["frontier"],
            current=current_state["current"],
            path=current_state["path"],
        ),
        use_container_width=True,
    )
    
    status_container.caption(
        f"Step {st.session_state.step_idx} / {max_steps} · "
        f"Current cell: {current_state['current']} · "
        f"Queue size: {current_state['queue_size']}"
    )
    metrics_container.metric("Maximum queue size seen", current_state["max_queue_size"])

    # 4. Handle Continuous Playback Loop via st.rerun()
    if st.session_state.playing:
        if st.session_state.step_idx < max_steps:
            time.sleep(delay)
            st.session_state.step_idx += 1
            st.rerun()
        else:
            st.session_state.playing = False
