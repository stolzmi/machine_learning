#!/usr/bin/env python3
"""
Swiss System Chess Tournament Manager
A complete implementation of Swiss-system tournament pairing and management.
"""

import random
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass, field


@dataclass
class Player:
    """Represents a player in the tournament."""
    id: int
    name: str
    score: float = 0.0
    opponents: Set[int] = field(default_factory=set)
    color_balance: int = 0  # Positive = more whites, Negative = more blacks
    
    def __repr__(self):
        return f"{self.name} (Score: {self.score})"


@dataclass
class Match:
    """Represents a match between two players."""
    white_id: int
    black_id: int
    result: str = "pending"  # "1-0", "0-1", "0.5-0.5", or "pending"
    
    def __repr__(self):
        return f"White: {self.white_id} vs Black: {self.black_id} [{self.result}]"


class SwissTournament:
    """Manages a Swiss-system chess tournament."""
    
    def __init__(self, name: str = "Chess Tournament"):
        self.name = name
        self.players: Dict[int, Player] = {}
        self.rounds: List[List[Match]] = []
        self.current_round = 0
        self.next_player_id = 1
        
    def add_player(self, name: str) -> int:
        """Add a player to the tournament."""
        player_id = self.next_player_id
        self.players[player_id] = Player(id=player_id, name=name)
        self.next_player_id += 1
        return player_id
    
    def add_players(self, names: List[str]) -> List[int]:
        """Add multiple players at once."""
        return [self.add_player(name) for name in names]
    
    def _get_score_groups(self) -> Dict[float, List[Player]]:
        """Group players by their current score."""
        score_groups = {}
        for player in self.players.values():
            if player.score not in score_groups:
                score_groups[player.score] = []
            score_groups[player.score].append(player)
        return score_groups
    
    def _can_pair(self, p1: Player, p2: Player) -> bool:
        """Check if two players can be paired (haven't played before)."""
        return p2.id not in p1.opponents and p1.id not in p2.opponents
    
    def _assign_colors(self, p1: Player, p2: Player) -> Tuple[int, int]:
        """
        Assign colors to two players based on color balance.
        Returns (white_id, black_id).
        """
        # Player with more blacks should get white
        if p1.color_balance < p2.color_balance:
            return (p1.id, p2.id)
        elif p2.color_balance < p1.color_balance:
            return (p2.id, p1.id)
        else:
            # Equal balance - randomize
            if random.random() < 0.5:
                return (p1.id, p2.id)
            else:
                return (p2.id, p1.id)
    
    def _pair_score_group(self, players: List[Player], unpaired: List[Player]) -> List[Match]:
        """
        Pair players within a score group using a greedy algorithm.
        Unpaired players are added to the unpaired list to be paired with lower groups.
        """
        matches = []
        available = players.copy()
        
        # Sort by rating/ID for consistency (in real tournaments, use rating)
        available.sort(key=lambda p: p.id)
        
        while len(available) >= 2:
            p1 = available[0]
            paired = False
            
            # Try to pair with someone they haven't played
            for i in range(1, len(available)):
                p2 = available[i]
                if self._can_pair(p1, p2):
                    white_id, black_id = self._assign_colors(p1, p2)
                    matches.append(Match(white_id=white_id, black_id=black_id))
                    available.remove(p1)
                    available.remove(p2)
                    paired = True
                    break
            
            if not paired:
                # Can't pair p1 in this group, move to unpaired list
                unpaired.append(p1)
                available.remove(p1)
        
        # Add any remaining player to unpaired list
        unpaired.extend(available)
        
        return matches
    
    def generate_pairings(self) -> List[Match]:
        """
        Generate pairings for the next round using Swiss system rules.
        """
        if not self.rounds or all(m.result != "pending" for m in self.rounds[-1]):
            self.current_round += 1
        else:
            raise ValueError("Cannot generate new pairings while current round is incomplete")
        
        # Get score groups (highest to lowest)
        score_groups = self._get_score_groups()
        sorted_scores = sorted(score_groups.keys(), reverse=True)
        
        matches = []
        unpaired = []
        
        # Pair each score group
        for score in sorted_scores:
            group_players = score_groups[score] + unpaired
            unpaired = []
            group_matches = self._pair_score_group(group_players, unpaired)
            matches.extend(group_matches)
        
        # Handle bye if odd number of players
        if unpaired:
            if len(unpaired) == 1:
                player = unpaired[0]
                # Give bye (1 point, represented as playing against ID 0)
                matches.append(Match(white_id=player.id, black_id=0, result="1-0"))
            else:
                # This shouldn't happen with proper pairing algorithm
                print(f"Warning: {len(unpaired)} players remain unpaired!")
        
        self.rounds.append(matches)
        return matches
    
    def record_result(self, round_num: int, match_num: int, result: str):
        """
        Record the result of a match.
        result should be: "1-0" (white wins), "0-1" (black wins), or "0.5-0.5" (draw)
        """
        if result not in ["1-0", "0-1", "0.5-0.5"]:
            raise ValueError("Result must be '1-0', '0-1', or '0.5-0.5'")
        
        match = self.rounds[round_num - 1][match_num]
        match.result = result
        
        white_id = match.white_id
        black_id = match.black_id
        
        # Update scores
        if result == "1-0":
            self.players[white_id].score += 1.0
            if black_id != 0:  # Not a bye
                self.players[black_id].score += 0.0
        elif result == "0-1":
            self.players[white_id].score += 0.0
            if black_id != 0:
                self.players[black_id].score += 1.0
        else:  # Draw
            self.players[white_id].score += 0.5
            if black_id != 0:
                self.players[black_id].score += 0.5
        
        # Update opponents and color balance
        if black_id != 0:  # Not a bye
            self.players[white_id].opponents.add(black_id)
            self.players[black_id].opponents.add(white_id)
            self.players[white_id].color_balance += 1
            self.players[black_id].color_balance -= 1
    
    def get_standings(self) -> List[Player]:
        """Get current standings sorted by score."""
        return sorted(self.players.values(), key=lambda p: p.score, reverse=True)
    
    def display_pairings(self, round_num: int = None):
        """Display pairings for a specific round."""
        if round_num is None:
            round_num = len(self.rounds)
        
        if round_num < 1 or round_num > len(self.rounds):
            print("Invalid round number")
            return
        
        print(f"\n{'='*60}")
        print(f"ROUND {round_num} PAIRINGS")
        print(f"{'='*60}")
        
        matches = self.rounds[round_num - 1]
        for i, match in enumerate(matches):
            white = self.players[match.white_id]
            if match.black_id == 0:
                print(f"Match {i+1}: {white.name} (BYE)")
            else:
                black = self.players[match.black_id]
                status = f"[{match.result}]" if match.result != "pending" else "[Not played]"
                print(f"Match {i+1}: {white.name} (W) vs {black.name} (B) {status}")
        print()
    
    def display_standings(self):
        """Display current tournament standings."""
        print(f"\n{'='*60}")
        print(f"TOURNAMENT STANDINGS - After Round {len(self.rounds)}")
        print(f"{'='*60}")
        print(f"{'Rank':<6} {'Name':<25} {'Score':<8} {'Played':<8}")
        print(f"{'-'*60}")
        
        standings = self.get_standings()
        for rank, player in enumerate(standings, 1):
            print(f"{rank:<6} {player.name:<25} {player.score:<8} {len(player.opponents):<8}")
        print()


def main():
    """Example usage of the Swiss Tournament system."""
    
    # Create tournament
    tournament = SwissTournament("Club Championship 2024")
    
    # Add players
    players = [
        "Alice Smith",
        "Bob Johnson", 
        "Carol Williams",
        "David Brown",
        "Eve Davis",
        "Frank Miller",
        "Grace Wilson",
        "Henry Moore"
    ]
    
    tournament.add_players(players)
    
    print(f"{'='*60}")
    print(f"{tournament.name}")
    print(f"{'='*60}")
    print(f"Number of players: {len(tournament.players)}")
    print(f"Players: {', '.join(players)}")
    
    # Simulate 3 rounds
    num_rounds = 3
    
    for round_num in range(1, num_rounds + 1):
        # Generate pairings
        tournament.generate_pairings()
        tournament.display_pairings(round_num)
        
        # Simulate random results for demonstration
        print(f"Recording results for Round {round_num}...")
        matches = tournament.rounds[round_num - 1]
        
        for match_idx, match in enumerate(matches):
            # Randomly assign results for demo purposes
            results = ["1-0", "0-1", "0.5-0.5"]
            if match.black_id == 0:  # Bye
                result = "1-0"
            else:
                result = random.choice(results)
            
            tournament.record_result(round_num, match_idx, result)
        
        # Display standings after round
        tournament.display_standings()
    
    print(f"\n{'='*60}")
    print(f"TOURNAMENT COMPLETE!")
    print(f"{'='*60}")
    
    standings = tournament.get_standings()
    winner = standings[0]
    print(f"\nCONGRATULATIONS to {winner.name}!")
    print(f"Final Score: {winner.score} points")


if __name__ == "__main__":
    main()