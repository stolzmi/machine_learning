#!/usr/bin/env python3
"""
FastAPI backend for Swiss System Chess Tournament Manager
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

from test import SwissTournament, Player, Match

app = FastAPI(title="Swiss Tournament Manager API")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global tournament instance
tournament = SwissTournament("Chess Tournament")


# Pydantic models for API
class PlayerCreate(BaseModel):
    name: str


class PlayerResponse(BaseModel):
    id: int
    name: str
    score: float
    opponents: List[int]
    color_balance: int


class MatchResponse(BaseModel):
    match_num: int
    white_id: int
    white_name: str
    black_id: int
    black_name: str
    result: str


class ResultRecord(BaseModel):
    result: str  # "1-0", "0-1", or "0.5-0.5"


class TournamentInfo(BaseModel):
    name: str
    total_players: int
    current_round: int
    total_rounds: int


@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Swiss Tournament Manager API", "status": "running"}


@app.post("/tournament/reset")
def reset_tournament(name: str = "Chess Tournament"):
    """Reset the tournament"""
    global tournament
    tournament = SwissTournament(name)
    return {"message": "Tournament reset successfully", "name": name}


@app.get("/tournament/info", response_model=TournamentInfo)
def get_tournament_info():
    """Get tournament information"""
    return TournamentInfo(
        name=tournament.name,
        total_players=len(tournament.players),
        current_round=tournament.current_round,
        total_rounds=len(tournament.rounds)
    )


@app.post("/players/", response_model=PlayerResponse)
def add_player(player: PlayerCreate):
    """Add a new player to the tournament"""
    player_id = tournament.add_player(player.name)
    p = tournament.players[player_id]
    return PlayerResponse(
        id=p.id,
        name=p.name,
        score=p.score,
        opponents=list(p.opponents),
        color_balance=p.color_balance
    )


@app.get("/players/", response_model=List[PlayerResponse])
def get_players():
    """Get all players"""
    return [
        PlayerResponse(
            id=p.id,
            name=p.name,
            score=p.score,
            opponents=list(p.opponents),
            color_balance=p.color_balance
        )
        for p in tournament.players.values()
    ]


@app.get("/standings/", response_model=List[PlayerResponse])
def get_standings():
    """Get current standings"""
    standings = tournament.get_standings()
    return [
        PlayerResponse(
            id=p.id,
            name=p.name,
            score=p.score,
            opponents=list(p.opponents),
            color_balance=p.color_balance
        )
        for p in standings
    ]


@app.post("/rounds/generate", response_model=List[MatchResponse])
def generate_round():
    """Generate pairings for the next round"""
    try:
        matches = tournament.generate_pairings()
        round_num = tournament.current_round

        return [
            MatchResponse(
                match_num=i,
                white_id=m.white_id,
                white_name=tournament.players[m.white_id].name,
                black_id=m.black_id,
                black_name="BYE" if m.black_id == 0 else tournament.players[m.black_id].name,
                result=m.result
            )
            for i, m in enumerate(matches)
        ]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/rounds/{round_num}", response_model=List[MatchResponse])
def get_round(round_num: int):
    """Get matches for a specific round"""
    if round_num < 1 or round_num > len(tournament.rounds):
        raise HTTPException(status_code=404, detail="Round not found")

    matches = tournament.rounds[round_num - 1]
    return [
        MatchResponse(
            match_num=i,
            white_id=m.white_id,
            white_name=tournament.players[m.white_id].name,
            black_id=m.black_id,
            black_name="BYE" if m.black_id == 0 else tournament.players[m.black_id].name,
            result=m.result
        )
        for i, m in enumerate(matches)
    ]


@app.get("/rounds/", response_model=Dict[int, List[MatchResponse]])
def get_all_rounds():
    """Get all rounds"""
    all_rounds = {}
    for round_num in range(1, len(tournament.rounds) + 1):
        matches = tournament.rounds[round_num - 1]
        all_rounds[round_num] = [
            MatchResponse(
                match_num=i,
                white_id=m.white_id,
                white_name=tournament.players[m.white_id].name,
                black_id=m.black_id,
                black_name="BYE" if m.black_id == 0 else tournament.players[m.black_id].name,
                result=m.result
            )
            for i, m in enumerate(matches)
        ]
    return all_rounds


@app.post("/rounds/{round_num}/matches/{match_num}/result")
def record_result(round_num: int, match_num: int, result_data: ResultRecord):
    """Record the result of a match"""
    try:
        tournament.record_result(round_num, match_num, result_data.result)
        return {"message": "Result recorded successfully"}
    except (ValueError, IndexError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/players/{player_id}")
def delete_player(player_id: int):
    """Delete a player (only if tournament hasn't started)"""
    if len(tournament.rounds) > 0:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete players after tournament has started"
        )

    if player_id not in tournament.players:
        raise HTTPException(status_code=404, detail="Player not found")

    del tournament.players[player_id]
    return {"message": "Player deleted successfully"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
