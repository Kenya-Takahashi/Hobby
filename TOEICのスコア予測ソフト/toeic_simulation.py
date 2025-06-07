import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from dataclasses import dataclass
import math

plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']

class EloRating:
    """Elo-Ratingシステムの実装"""
    
    def __init__(self, k_factor: int = 32):
        self.k_factor = k_factor
    
    def calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """期待スコアを計算する"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_rating(self, current_rating: float, expected_score: float, actual_score: float) -> float:
        """レーティングを更新する"""
        return current_rating + self.k_factor * (actual_score - expected_score)

@dataclass
class TOEICTestConfig:
    """TOEICテストの構成情報"""
    listening_parts = {
        'part1': 6,   # 写真描写問題
        'part2': 25,  # 応答問題
        'part3': 39,  # 会話問題
        'part4': 30   # 説明文問題
    }
    
    reading_parts = {
        'part5': 30,  # 短文穴埋め問題
        'part6': 16,  # 長文穴埋め問題
        'part7': 54   # 文書問題
    }
    
    @property
    def total_questions(self) -> int:
        """総問題数を返す"""
        return sum(self.listening_parts.values()) + sum(self.reading_parts.values())
    
    @property
    def listening_questions(self) -> int:
        """リスニング問題数を返す"""
        return sum(self.listening_parts.values())
    
    @property
    def reading_questions(self) -> int:
        """リーディング問題数を返す"""
        return sum(self.reading_parts.values())

class UserAgent:
    """ユーザーエージェント"""
    
    def __init__(self, initial_rating: float):
        self.rating = initial_rating
        self.elo_system = EloRating()
    
    def solve_question(self, question_rating: float) -> bool:
        """問題を解く（確率的に正解/不正解を決定）"""
        expected_score = self.elo_system.calculate_expected_score(self.rating, question_rating)
        return random.random() < expected_score
    
    def update_rating_after_test(self, question_rating: float, correct_answers: int, total_questions: int):
        """テスト後にレーティングを更新"""
        actual_score = correct_answers / total_questions
        expected_score = self.elo_system.calculate_expected_score(self.rating, question_rating)
        self.rating = self.elo_system.update_rating(self.rating, expected_score, actual_score)

class QuestionAgent:
    """問題エージェント（レーティング固定）"""
    
    def __init__(self, rating: float = 1500):
        self.rating = rating

class TOEICSimulation:
    """TOEICシミュレーション実行クラス"""
    
    def __init__(self):
        self.config = TOEICTestConfig()
        self.question_agent = QuestionAgent()
    
    def run_single_test(self, user_agent: UserAgent) -> Tuple[int, int]:
        """1回のテストを実行"""
        correct_answers = 0
        total_questions = self.config.total_questions
        
        for _ in range(total_questions):
            if user_agent.solve_question(self.question_agent.rating):
                correct_answers += 1
        
        score = correct_answers * 5
        user_agent.update_rating_after_test(self.question_agent.rating, correct_answers, total_questions)
        
        return correct_answers, score
    
    def run_simulation(self, initial_rating: float, num_trials: int = 1000) -> Dict:
        """指定回数のシミュレーションを実行"""
        results = {
            'scores': [],
            'correct_answers': [],
            'rating_history': []
        }
        
        user_agent = UserAgent(initial_rating)
        results['rating_history'].append(user_agent.rating)
        
        for trial in range(num_trials):
            correct, score = self.run_single_test(user_agent)
            results['correct_answers'].append(correct)
            results['scores'].append(score)
            results['rating_history'].append(user_agent.rating)
        
        return results
    
    def run_multi_agent_simulation(self, initial_ratings: List[float], num_trials: int = 1000) -> Dict:
        """複数の初期レーティングでMASを実行"""
        all_results = {}
        
        for rating in initial_ratings:
            print(f"初期レーティング {rating} でシミュレーション実行中...")
            results = self.run_simulation(rating, num_trials)
            all_results[rating] = results
        
        return all_results

class ResultAnalyzer:
    """結果分析・可視化クラス"""
    
    @staticmethod
    def plot_score_distribution(results: Dict, initial_ratings: List[float]):
        """スコア分布を可視化"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, rating in enumerate(initial_ratings):
            scores = results[rating]['scores']
            axes[i].hist(scores, bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Initial Rating: {rating}\nAverage Score: {np.mean(scores):.1f}')
            axes[i].set_xlabel('Score')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        # 最後のサブプロットを削除（7個しかないため）
        fig.delaxes(axes[7])
        
        plt.tight_layout()
        plt.suptitle('TOEIC Score Distribution (1000 trials)', fontsize=16, y=1.02)
        plt.savefig('toeic_score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_rating_evolution(results: Dict, initial_ratings: List[float]):
        """レーティングの変化を可視化"""
        plt.figure(figsize=(15, 8))
        
        for rating in initial_ratings:
            rating_history = results[rating]['rating_history']
            plt.plot(rating_history, label=f'Initial Rating: {rating}', alpha=0.8)
        
        plt.axhline(y=1500, color='red', linestyle='--', alpha=0.7, label='Question Rating: 1500')
        plt.xlabel('Trial Number')
        plt.ylabel('Rating')
        plt.title('Rating Evolution (1000 trials)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('rating_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def print_statistics(results: Dict, initial_ratings: List[float]):
        """統計情報を表示"""
        print("\n=== シミュレーション結果統計 ===")
        print(f"{'初期レーティング':<12} {'平均スコア':<10} {'標準偏差':<10} {'最終レーティング':<12} {'収束値':<10}")
        print("-" * 65)
        
        for rating in initial_ratings:
            scores = results[rating]['scores']
            final_rating = results[rating]['rating_history'][-1]
            last_100_ratings = results[rating]['rating_history'][-100:]
            convergence = np.mean(last_100_ratings)
            
            print(f"{rating:<12} {np.mean(scores):<10.1f} {np.std(scores):<10.1f} "
                  f"{final_rating:<12.1f} {convergence:<10.1f}")

def main():
    """メイン実行関数"""
    # 初期レーティング設定
    initial_ratings = [1200, 1300, 1400, 1500, 1600, 1700, 1800]
    
    # シミュレーションの実行
    simulation = TOEICSimulation()
    results = simulation.run_multi_agent_simulation(initial_ratings, num_trials=1000)
    
    # 結果の分析・可視化
    analyzer = ResultAnalyzer()
    analyzer.print_statistics(results, initial_ratings)
    analyzer.plot_score_distribution(results, initial_ratings)
    analyzer.plot_rating_evolution(results, initial_ratings)

if __name__ == "__main__":
    main()