#!/usr/bin/env python3
"""
Streamlit frontend for Swiss System Chess Tournament Manager
"""

import streamlit as st
import requests
from typing import List, Dict
import pandas as pd

# Backend API URL
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Swiss Tournament Manager",
    page_icon="♟️",
    layout="wide"
)


def init_session_state():
    """Initialize session state variables"""
    if 'current_round' not in st.session_state:
        st.session_state.current_round = 0


def get_tournament_info():
    """Get tournament information from API"""
    try:
        response = requests.get(f"{API_URL}/tournament/info")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None


def add_player(name: str):
    """Add a player via API"""
    try:
        response = requests.post(f"{API_URL}/players/", json={"name": name})
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error adding player: {e}")
        return False


def get_players():
    """Get all players from API"""
    try:
        response = requests.get(f"{API_URL}/players/")
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Error getting players: {e}")
        return []


def get_standings():
    """Get current standings from API"""
    try:
        response = requests.get(f"{API_URL}/standings/")
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Error getting standings: {e}")
        return []


def generate_round():
    """Generate new round pairings"""
    try:
        response = requests.post(f"{API_URL}/rounds/generate")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error generating round: {e}")
        return None


def get_round(round_num: int):
    """Get matches for a specific round"""
    try:
        response = requests.get(f"{API_URL}/rounds/{round_num}")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error getting round: {e}")
        return None


def record_result(round_num: int, match_num: int, result: str):
    """Record match result"""
    try:
        response = requests.post(
            f"{API_URL}/rounds/{round_num}/matches/{match_num}/result",
            json={"result": result}
        )
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error recording result: {e}")
        return False


def reset_tournament(name: str):
    """Reset the tournament"""
    try:
        response = requests.post(f"{API_URL}/tournament/reset", params={"name": name})
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error resetting tournament: {e}")
        return False


def main():
    init_session_state()

    st.title("♟️ Swiss System Tournament Manager")

    # Sidebar for tournament setup
    with st.sidebar:
        st.header("Tournament Setup")

        tournament_info = get_tournament_info()

        if tournament_info:
            st.info(f"**Tournament:** {tournament_info['name']}")
            st.info(f"**Players:** {tournament_info['total_players']}")
            st.info(f"**Current Round:** {tournament_info['current_round']}")

        st.subheader("Reset Tournament")
        new_tournament_name = st.text_input("Tournament Name", value="Chess Tournament")
        if st.button("Reset Tournament", type="secondary"):
            if reset_tournament(new_tournament_name):
                st.success("Tournament reset!")
                st.rerun()

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Players", "Generate Round", "Record Results", "Standings"])

    # Tab 1: Add Players
    with tab1:
        st.header("Manage Players")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Add New Player")
            player_name = st.text_input("Player Name", key="player_name")

            if st.button("Add Player", type="primary"):
                if player_name:
                    if add_player(player_name):
                        st.success(f"Added {player_name}!")
                        st.rerun()
                else:
                    st.warning("Please enter a player name")

            st.subheader("Bulk Add Players")
            bulk_names = st.text_area(
                "Enter player names (one per line)",
                height=150,
                placeholder="Alice Smith\nBob Johnson\nCarol Williams"
            )

            if st.button("Add All Players", type="primary"):
                if bulk_names:
                    names = [name.strip() for name in bulk_names.split('\n') if name.strip()]
                    success_count = 0
                    for name in names:
                        if add_player(name):
                            success_count += 1
                    st.success(f"Added {success_count} players!")
                    st.rerun()

        with col2:
            st.subheader("Current Players")
            players = get_players()
            if players:
                for player in players:
                    st.text(f"{player['id']}. {player['name']}")
            else:
                st.info("No players added yet")

    # Tab 2: Generate Round
    with tab2:
        st.header("Generate Round Pairings")

        tournament_info = get_tournament_info()

        if tournament_info and tournament_info['total_players'] >= 2:
            if st.button("Generate Next Round", type="primary", key="generate_btn"):
                matches = generate_round()
                if matches:
                    st.success(f"Round {tournament_info['current_round'] + 1} generated!")
                    st.rerun()
        else:
            st.warning("Add at least 2 players before generating rounds")

        # Show current round pairings
        if tournament_info and tournament_info['current_round'] > 0:
            st.subheader(f"Round {tournament_info['current_round']} Pairings")
            matches = get_round(tournament_info['current_round'])

            if matches:
                for match in matches:
                    col1, col2, col3, col4 = st.columns([2, 1, 2, 2])
                    with col1:
                        st.write(f"**{match['white_name']}**")
                    with col2:
                        st.write("🆚")
                    with col3:
                        st.write(f"**{match['black_name']}**")
                    with col4:
                        result_text = match['result'] if match['result'] != 'pending' else 'Not played'
                        st.write(f"Result: {result_text}")

    # Tab 3: Record Results
    with tab3:
        st.header("Record Match Results")

        tournament_info = get_tournament_info()

        if tournament_info and tournament_info['current_round'] > 0:
            # Select round
            round_num = st.selectbox(
                "Select Round",
                range(1, tournament_info['current_round'] + 1),
                index=tournament_info['current_round'] - 1
            )

            matches = get_round(round_num)

            if matches:
                st.subheader(f"Round {round_num} Matches")

                for match in matches:
                    with st.container():
                        col1, col2, col3 = st.columns([3, 2, 1])

                        with col1:
                            match_text = f"**Match {match['match_num'] + 1}:** {match['white_name']} (White) vs {match['black_name']} (Black)"
                            st.write(match_text)

                        with col2:
                            result_key = f"result_{round_num}_{match['match_num']}"
                            result = st.selectbox(
                                "Result",
                                ["Select...", "1-0 (White wins)", "0-1 (Black wins)", "0.5-0.5 (Draw)"],
                                key=result_key,
                                label_visibility="collapsed"
                            )

                        with col3:
                            if st.button("Submit", key=f"submit_{round_num}_{match['match_num']}"):
                                if result != "Select...":
                                    result_code = result.split()[0]
                                    if record_result(round_num, match['match_num'], result_code):
                                        st.success("Result recorded!")
                                        st.rerun()
                                else:
                                    st.warning("Please select a result")

                        # Show current result
                        if match['result'] != 'pending':
                            st.caption(f"✓ Current result: {match['result']}")

                        st.divider()
        else:
            st.info("No rounds generated yet. Go to 'Generate Round' tab to start.")

    # Tab 4: Standings
    with tab4:
        st.header("Tournament Standings")

        standings = get_standings()

        if standings:
            # Create DataFrame for better display
            standings_data = []
            for rank, player in enumerate(standings, 1):
                standings_data.append({
                    "Rank": rank,
                    "Name": player['name'],
                    "Score": player['score'],
                    "Matches Played": len(player['opponents']),
                    "Color Balance": player['color_balance']
                })

            df = pd.DataFrame(standings_data)

            # Style the dataframe
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn(format="%d"),
                    "Score": st.column_config.NumberColumn(format="%.1f"),
                    "Matches Played": st.column_config.NumberColumn(format="%d"),
                    "Color Balance": st.column_config.NumberColumn(
                        format="%d",
                        help="Positive = more games as White, Negative = more games as Black"
                    )
                }
            )

            # Show winner if tournament has matches
            tournament_info = get_tournament_info()
            if tournament_info and tournament_info['current_round'] > 0:
                winner = standings_data[0]
                st.success(f"🏆 Current Leader: **{winner['Name']}** with {winner['Score']} points!")
        else:
            st.info("No players in tournament yet")


if __name__ == "__main__":
    main()
