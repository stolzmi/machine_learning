# Swiss System Tournament Manager

A complete tournament management system with FastAPI backend and Streamlit frontend.

## Files

- `test.py` - Core tournament logic (Swiss system implementation)
- `backend.py` - FastAPI REST API server
- `frontend.py` - Streamlit web interface
- `requirements.txt` - Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Running the Application

### Step 1: Start the Backend API

In one terminal:

```bash
python backend.py
```

The API will run on http://localhost:8000

### Step 2: Start the Frontend

In another terminal:

```bash
streamlit run frontend.py
```

The web interface will open in your browser (usually http://localhost:8501)

## Features

### Players Tab
- Add individual players
- Bulk add multiple players at once
- View all registered players

### Generate Round Tab
- Generate Swiss system pairings
- View current round matches
- Automatic bye handling for odd number of players

### Record Results Tab
- Select any round
- Record match results (1-0, 0-1, 0.5-0.5)
- Update results for completed matches

### Standings Tab
- View real-time tournament standings
- See player scores, matches played, and color balance
- Identify current tournament leader

## API Endpoints

- `GET /` - API status
- `POST /tournament/reset` - Reset tournament
- `GET /tournament/info` - Get tournament info
- `POST /players/` - Add player
- `GET /players/` - Get all players
- `GET /standings/` - Get current standings
- `POST /rounds/generate` - Generate next round
- `GET /rounds/{round_num}` - Get round matches
- `POST /rounds/{round_num}/matches/{match_num}/result` - Record result

## Swiss System Rules

The system implements standard Swiss tournament rules:
- Players with equal scores are paired when possible
- Players don't face the same opponent twice
- Color balance is maintained (alternating white/black)
- Automatic bye for odd number of players
