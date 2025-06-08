import matplotlib.pyplot as plt
import seaborn as sns
from tournament_simulator import Simulator, create_sample_players
import pandas as pd


def create_win_probability_chart(results_df: pd.DataFrame, title: str = "Tournament Win Probability"):
    """優勝確率の棒グラフを作成"""
    plt.figure(figsize=(12, 8))
    
    # 棒グラフを作成
    bars = plt.bar(results_df['プレイヤー名'], results_df['優勝確率（数値）'], 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    
    # グラフの装飾
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Player', fontsize=12)
    plt.ylabel('Win Probability', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # 各バーの上に値を表示
    for bar, prob in zip(bars, results_df['優勝確率（数値）']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{prob:.1%}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    return plt


def create_rating_vs_probability_scatter(results_df: pd.DataFrame):
    """レーティングと優勝確率の散布図を作成"""
    plt.figure(figsize=(10, 8))
    
    plt.scatter(results_df['Eloレーティング'], results_df['優勝確率（数値）'], 
               s=100, color='red', alpha=0.7, edgecolors='darkred')
    
    # プレイヤー名をラベルとして追加
    for _, row in results_df.iterrows():
        plt.annotate(row['プレイヤー名'], 
                    (row['Eloレーティング'], row['優勝確率（数値）']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.title('Elo Rating vs Win Probability', fontsize=16, fontweight='bold')
    plt.xlabel('Elo Rating', fontsize=12)
    plt.ylabel('Win Probability', fontsize=12)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    plt.tight_layout()
    plt.grid(alpha=0.3)
    return plt


def main():
    """可視化のデモ実行"""
    print("=== 可視化デモ実行中 ===\n")
    
    # サンプルプレイヤーでシミュレーション実行
    players = create_sample_players()
    simulator = Simulator(players)
    
    print("シミュレーション実行中...")
    simulator.run_simulation(num_simulations=5000)
    results_df = simulator.get_results_dataframe()
    
    # 優勝確率の棒グラフを作成・保存
    print("優勝確率の棒グラフを作成中...")
    plt1 = create_win_probability_chart(results_df)
    plt1.savefig('win_probability_chart.png', dpi=300, bbox_inches='tight')
    print("win_probability_chart.png に保存しました。")
    plt1.show()
    
    # レーティングvs確率の散布図を作成・保存
    print("\nレーティングvs確率の散布図を作成中...")
    plt2 = create_rating_vs_probability_scatter(results_df)
    plt2.savefig('rating_vs_probability.png', dpi=300, bbox_inches='tight')
    print("rating_vs_probability.png に保存しました。")
    plt2.show()
    
    print("\n可視化完了！")


if __name__ == "__main__":
    main()