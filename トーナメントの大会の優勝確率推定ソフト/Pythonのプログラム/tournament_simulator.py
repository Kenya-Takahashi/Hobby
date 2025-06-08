import random
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm


@dataclass
class Player:
    name: str
    rating: float
    
    def __post_init__(self):
        self.wins = 0
        self.losses = 0
        self.tournaments_won = 0


class EloRating:
    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """プレイヤーAの期待勝率を計算"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    @staticmethod
    def update_rating(rating: float, expected: float, actual: float, k_factor: float = 32) -> float:
        """Eloレーティングの更新"""
        return rating + k_factor * (actual - expected)


class Match:
    def __init__(self, player1: Player, player2: Player):
        self.player1 = player1
        self.player2 = player2
        self.winner: Optional[Player] = None
        self.loser: Optional[Player] = None
    
    def simulate(self, update_ratings: bool = False) -> Player:
        """試合をシミュレートし、勝者を返す"""
        expected_score = EloRating.expected_score(self.player1.rating, self.player2.rating)
        
        # 確率的に勝者を決定
        if random.random() < expected_score:
            self.winner = self.player1
            self.loser = self.player2
        else:
            self.winner = self.player2
            self.loser = self.player1
        
        # 戦績更新
        self.winner.wins += 1
        self.loser.losses += 1
        
        # Eloレーティング更新（オプション）
        if update_ratings:
            if self.winner == self.player1:
                actual_score_p1, actual_score_p2 = 1.0, 0.0
            else:
                actual_score_p1, actual_score_p2 = 0.0, 1.0
            
            new_rating_p1 = EloRating.update_rating(
                self.player1.rating, expected_score, actual_score_p1
            )
            new_rating_p2 = EloRating.update_rating(
                self.player2.rating, 1 - expected_score, actual_score_p2
            )
            
            self.player1.rating = new_rating_p1
            self.player2.rating = new_rating_p2
        
        return self.winner


class Tournament:
    def __init__(self, players: List[Player], tournament_type: str = "single_elimination"):
        self.players = players.copy()
        self.tournament_type = tournament_type
        self.bracket = []
        self.results = {}
        self.winner: Optional[Player] = None
        
        # プレイヤー数が2の累乗になるようにダミープレイヤーを追加
        self._pad_players()
    
    def _pad_players(self):
        """プレイヤー数を2の累乗に調整"""
        n = len(self.players)
        next_power = 2 ** math.ceil(math.log2(n))
        
        while len(self.players) < next_power:
            # ダミープレイヤーを追加（レーティング0で自動的に負ける）
            dummy = Player(f"BYE_{len(self.players)}", 0)
            self.players.append(dummy)
    
    def simulate_single_elimination(self, update_ratings: bool = False) -> Player:
        """シングルエリミネーション トーナメントをシミュレート"""
        current_round = self.players.copy()
        round_num = 1
        
        while len(current_round) > 1:
            next_round = []
            
            # ペアを作って対戦
            for i in range(0, len(current_round), 2):
                player1 = current_round[i]
                player2 = current_round[i + 1]
                
                # BYEプレイヤーの場合は自動勝利
                if player2.name.startswith("BYE_"):
                    winner = player1
                elif player1.name.startswith("BYE_"):
                    winner = player2
                else:
                    match = Match(player1, player2)
                    winner = match.simulate(update_ratings)
                
                next_round.append(winner)
            
            current_round = next_round
            round_num += 1
        
        self.winner = current_round[0]
        if not self.winner.name.startswith("BYE_"):
            self.winner.tournaments_won += 1
        
        return self.winner
    
    def simulate(self, update_ratings: bool = False) -> Player:
        """指定された形式でトーナメントをシミュレート"""
        if self.tournament_type == "single_elimination":
            return self.simulate_single_elimination(update_ratings)
        else:
            raise ValueError(f"Unsupported tournament type: {self.tournament_type}")


class Simulator:
    def __init__(self, players: List[Player]):
        self.original_players = players
        self.simulation_results = defaultdict(int)
        self.total_simulations = 0
    
    def reset_players(self) -> List[Player]:
        """プレイヤーの状態をリセット"""
        reset_players = []
        for player in self.original_players:
            new_player = Player(player.name, player.rating)
            reset_players.append(new_player)
        return reset_players
    
    def run_simulation(self, num_simulations: int = 1000, 
                      tournament_type: str = "single_elimination",
                      update_ratings: bool = False) -> Dict[str, float]:
        """複数回のトーナメントシミュレーションを実行"""
        self.simulation_results.clear()
        self.total_simulations = num_simulations
        
        for _ in tqdm(range(num_simulations), desc="シミュレーション実行中"):
            # プレイヤーをリセット
            players = self.reset_players()
            
            # トーナメント実行
            tournament = Tournament(players, tournament_type)
            winner = tournament.simulate(update_ratings)
            
            # BYEプレイヤーでない場合のみカウント
            if not winner.name.startswith("BYE_"):
                self.simulation_results[winner.name] += 1
        
        # 優勝確率を計算
        win_probabilities = {}
        for player_name, wins in self.simulation_results.items():
            win_probabilities[player_name] = wins / num_simulations
        
        return win_probabilities
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """結果をDataFrameで返す"""
        results_data = []
        
        for player in self.original_players:
            wins = self.simulation_results.get(player.name, 0)
            probability = wins / self.total_simulations if self.total_simulations > 0 else 0
            
            results_data.append({
                'プレイヤー名': player.name,
                'Eloレーティング': player.rating,
                '優勝回数': wins,
                '優勝確率': f"{probability:.1%}",
                '優勝確率（数値）': probability
            })
        
        df = pd.DataFrame(results_data)
        df = df.sort_values('優勝確率（数値）', ascending=False).reset_index(drop=True)
        df['順位'] = range(1, len(df) + 1)
        
        # 可視化用にも優勝確率（数値）を含める
        return df[['順位', 'プレイヤー名', 'Eloレーティング', '優勝回数', '優勝確率', '優勝確率（数値）']]


def create_sample_players() -> List[Player]:
    """サンプルプレイヤーを作成"""
    players = [
        Player("Player1", 1800),
        Player("Player2", 1600),
        Player("Player3", 1550),
        Player("Player4", 1500),
        Player("Player5", 1450),
        Player("Player6", 1400),
        Player("Player7", 1350),
        Player("Player8", 1300),
    ]
    return players


def main():
    """メイン実行関数"""
    print("=== MAS & Elo Rating トーナメント優勝確率推定ソフト ===\n")
    
    # サンプルプレイヤーを作成
    players = create_sample_players()
    
    print("参加プレイヤー:")
    for i, player in enumerate(players, 1):
        print(f"{i}. {player.name} (レーティング: {player.rating})")
    
    print(f"\n{len(players)}人でシングルエリミネーション トーナメントをシミュレーションします。")
    
    # シミュレータを作成
    simulator = Simulator(players)
    
    # シミュレーション実行
    num_simulations = 10000
    print(f"\n{num_simulations:,}回のシミュレーションを実行します...")
    
    win_probabilities = simulator.run_simulation(
        num_simulations=num_simulations,
        tournament_type="single_elimination"
    )
    
    # 結果表示
    print("\n=== シミュレーション結果 ===")
    results_df = simulator.get_results_dataframe()
    print(results_df.to_string(index=False))
    
    print(f"\n総シミュレーション回数: {num_simulations:,}回")
    print("※ レーティングが高いプレイヤーほど優勝確率が高くなる傾向があります。")


if __name__ == "__main__":
    main()