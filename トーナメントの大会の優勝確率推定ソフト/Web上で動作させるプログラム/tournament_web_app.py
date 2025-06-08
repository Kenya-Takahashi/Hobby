import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Pythonのプログラム'))

from tournament_simulator import Player, Simulator
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def create_sample_players():
    """サンプルプレイヤーを作成"""
    return [
        Player("Player1", 1800),
        Player("Player2", 1600),
        Player("Player3", 1550),
        Player("Player4", 1500),
        Player("Player5", 1450),
        Player("Player6", 1400),
        Player("Player7", 1350),
        Player("Player8", 1300),
    ]


def create_win_probability_chart(results_df):
    """優勝確率の棒グラフを作成（Plotly版）"""
    fig = px.bar(
        results_df, 
        x='プレイヤー名', 
        y='優勝確率（数値）',
        title='Tournament Win Probability',
        labels={'プレイヤー名': 'Player', '優勝確率（数値）': 'Win Probability'},
        color='優勝確率（数値）',
        color_continuous_scale='Blues'
    )
    
    # パーセント表示に変更
    fig.update_yaxes(tickformat='.1%')
    
    # 各バーの上に値を表示
    fig.update_traces(
        texttemplate='%{y:.1%}',
        textposition='outside'
    )
    
    fig.update_layout(
        showlegend=False,
        height=500,
        xaxis_title="Player",
        yaxis_title="Win Probability"
    )
    
    return fig


def create_rating_vs_probability_scatter(results_df):
    """レーティングと優勝確率の散布図を作成（Plotly版）"""
    fig = px.scatter(
        results_df,
        x='Eloレーティング',
        y='優勝確率（数値）',
        text='プレイヤー名',
        title='Elo Rating vs Win Probability',
        labels={'Eloレーティング': 'Elo Rating', '優勝確率（数値）': 'Win Probability'},
        size='優勝確率（数値）',
        color='優勝確率（数値）',
        color_continuous_scale='Reds',
        size_max=15
    )
    
    # テキストの位置調整
    fig.update_traces(textposition="top center")
    
    # パーセント表示に変更
    fig.update_yaxes(tickformat='.1%')
    
    fig.update_layout(
        showlegend=False,
        height=500,
        xaxis_title="Elo Rating",
        yaxis_title="Win Probability"
    )
    
    return fig


def run_simulation(players, num_simulations):
    """シミュレーションを実行"""
    simulator = Simulator(players)
    
    # プログレスバー付きでシミュレーション実行
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 実際のシミュレーション実行
    win_probabilities = simulator.run_simulation(num_simulations=num_simulations)
    
    progress_bar.progress(100)
    status_text.text(f'Simulation completed! ({num_simulations:,} runs)')
    
    results_df = simulator.get_results_dataframe()
    return results_df


def main():
    """メインのStreamlitアプリ"""
    st.set_page_config(
        page_title="Tournament Win Probability Simulator",
        page_icon="🏆",
        layout="wide"
    )
    
    # ページ選択
    page = st.sidebar.selectbox(
        "Select Page",
        ["Main Simulation", "What-if Analysis"],
        key="page_selector"
    )
    
    if page == "Main Simulation":
        run_main_simulation()
    else:
        run_whatif_analysis()


def run_main_simulation():
    """メインのシミュレーションページ"""
    st.title("🏆 MAS & Elo Rating Tournament Simulator")
    st.markdown("**Multi-Agent Simulation with Elo Rating System**")
    
    # サイドバーで設定
    st.sidebar.header("⚙️ Tournament Settings")
    
    # プリセットまたはカスタム選択
    setting_type = st.sidebar.radio(
        "Player Configuration",
        ["Use Preset Players", "Custom Players"]
    )
    
    players = []
    
    if setting_type == "Use Preset Players":
        players = create_sample_players()
        st.sidebar.success(f"✅ Using {len(players)} preset players")
        
        # プリセットプレイヤーの表示
        st.sidebar.subheader("Preset Players:")
        for i, player in enumerate(players, 1):
            st.sidebar.write(f"{i}. {player.name} (Rating: {player.rating})")
    
    else:
        # カスタムプレイヤー設定
        st.sidebar.subheader("Custom Player Setup")
        
        num_players = st.sidebar.slider(
            "Number of Players",
            min_value=2,
            max_value=16,
            value=8,
            help="Select the number of participants"
        )
        
        st.sidebar.write("**Player Information:**")
        
        for i in range(num_players):
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                name = st.text_input(
                    f"Player {i+1} Name",
                    value=f"Player{i+1}",
                    key=f"name_{i}"
                )
            
            with col2:
                rating = st.number_input(
                    f"Rating",
                    min_value=800,
                    max_value=3000,
                    value=max(800, 1500 - i*50),
                    step=10,
                    key=f"rating_{i}"
                )
            
            players.append(Player(name, rating))
    
    # シミュレーション回数設定
    num_simulations = st.sidebar.slider(
        "Number of Simulations",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
        help="More simulations = more accurate results (but slower)"
    )
    
    # メインエリア
    if st.sidebar.button("🚀 Run Simulation", type="primary"):
        st.header("📊 Simulation Results")
        
        # シミュレーション実行
        with st.spinner('Running tournament simulations...'):
            results_df = run_simulation(players, num_simulations)
            
            # 結果をセッション状態に保存（What-if分析で使用）
            st.session_state.main_results = {
                'results': results_df,
                'players': players,
                'num_simulations': num_simulations
            }
        
        # 結果の表示
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Win Probability Ranking")
            
            # 結果テーブル（順位、名前、レーティング、確率のみ表示）
            display_df = results_df[['順位', 'プレイヤー名', 'Eloレーティング', '優勝確率']].copy()
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
        with col2:
            st.subheader("📈 Statistics")
            
            probabilities = results_df['優勝確率（数値）']
            
            # 統計情報をメトリクスで表示
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                st.metric("Highest Probability", f"{probabilities.max():.1%}")
                st.metric("Lowest Probability", f"{probabilities.min():.1%}")
            
            with col2_2:
                st.metric("Standard Deviation", f"{probabilities.std():.3f}")
                st.metric("Total Simulations", f"{num_simulations:,}")
        
        # グラフ表示
        st.header("📊 Visualization")
        
        tab1, tab2 = st.tabs(["📊 Win Probability Chart", "🎯 Rating vs Probability"])
        
        with tab1:
            fig1 = create_win_probability_chart(results_df)
            st.plotly_chart(fig1, use_container_width=True)
        
        with tab2:
            fig2 = create_rating_vs_probability_scatter(results_df)
            st.plotly_chart(fig2, use_container_width=True)
        
        # What-if分析は別ページで実行
        st.header("🔄 What-if Analysis")
        st.info("👈 Use the sidebar to switch to 'What-if Analysis' page for detailed scenario analysis!")
        
        # 簡単なWhat-if例を表示
        st.markdown("""
        **What-if Analysis allows you to:**
        - Change any player's rating
        - See how it affects everyone's win probability
        - Compare before/after scenarios
        - Analyze strategic impact
        
        **Example scenarios:**
        - "What if the strongest player doesn't participate?"
        - "How much would improving my rating help?"
        - "What's the impact of adding a new strong player?"
        """)
    
    else:
        # 初期画面
        st.header("👋 Welcome to Tournament Simulator!")
        st.markdown("""
        This application simulates tournament outcomes using:
        
        - **🎯 Elo Rating System**: Calculates win probabilities based on player ratings
        - **🤖 Multi-Agent Simulation**: Each player acts as an independent agent
        - **📊 Statistical Analysis**: Runs thousands of simulations for accurate predictions
        - **📈 Interactive Visualization**: Real-time charts and what-if analysis
        
        **How to use:**
        1. Configure players in the sidebar (preset or custom)
        2. Set the number of simulations
        3. Click "Run Simulation" to see results
        4. Explore what-if scenarios to see how rating changes affect outcomes
        """)
        
        # システムの仕組み説明
        st.subheader("🧮 How It Works")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Elo Rating Formula:**
            ```
            Expected Win Rate = 1 / (1 + 10^((Opponent Rating - Player Rating) / 400))
            ```
            
            **Tournament Structure:**
            - Single Elimination format
            - Automatic BYE placement for non-power-of-2 players
            - Supports 2-16 players
            """)
        
        with col2:
            st.markdown("""
            **Simulation Process:**
            1. Calculate expected win rates for each match
            2. Use probability to determine winners
            3. Repeat tournament thousands of times
            4. Calculate statistical win probabilities
            
            **Features:**
            - Real-time progress tracking
            - Interactive charts with Plotly
            - What-if analysis for strategy planning
            """)


def run_whatif_analysis():
    """What-if分析専用ページ"""
    st.title("🔄 What-if Analysis")
    st.markdown("**Analyze how rating changes affect win probabilities**")
    
    # メインシミュレーションの結果が必要
    if 'main_results' not in st.session_state:
        st.warning("⚠️ Please run the main simulation first!")
        st.markdown("1. Go to 'Main Simulation' page in the sidebar")
        st.markdown("2. Run a tournament simulation")
        st.markdown("3. Come back to this page for What-if analysis")
        return
    
    # メインの結果を取得
    main_data = st.session_state.main_results
    original_results = main_data['results']
    players = main_data['players']
    num_simulations = main_data['num_simulations']
    
    st.success(f"✅ Using results from main simulation ({num_simulations:,} runs)")
    
    # 元の結果を表示
    with st.expander("📊 Original Results", expanded=False):
        st.dataframe(original_results[['順位', 'プレイヤー名', 'Eloレーティング', '優勝確率']], use_container_width=True)
    
    # What-if設定
    st.header("⚙️ What-if Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_player_idx = st.selectbox(
            "Select Player to Modify",
            range(len(players)),
            format_func=lambda x: f"{players[x].name} (Current: {players[x].rating})"
        )
    
    with col2:
        new_rating = st.number_input(
            "New Rating",
            min_value=800,
            max_value=3000,
            value=int(players[selected_player_idx].rating),
            step=10
        )
    
    if st.button("🚀 Run What-if Analysis", type="primary"):
        # 新しいプレイヤーリストを作成
        modified_players = []
        for i, player in enumerate(players):
            if i == selected_player_idx:
                modified_players.append(Player(player.name, new_rating))
            else:
                modified_players.append(Player(player.name, player.rating))
        
        # What-if分析実行
        with st.spinner('Running what-if analysis...'):
            simulator = Simulator(modified_players)
            whatif_win_probabilities = simulator.run_simulation(num_simulations=num_simulations // 2)
            whatif_results = simulator.get_results_dataframe()
        
        # 結果表示
        st.header("📊 What-if Results")
        
        # 変更内容
        st.info(f"**Changed**: {players[selected_player_idx].name}'s rating from {players[selected_player_idx].rating} to {new_rating}")
        
        # 比較表
        comparison_data = []
        for _, row in original_results.iterrows():
            original_prob = row['優勝確率（数値）']
            player_name = row['プレイヤー名']
            
            whatif_row = whatif_results[whatif_results['プレイヤー名'] == player_name]
            if not whatif_row.empty:
                new_prob = whatif_row.iloc[0]['優勝確率（数値）']
                change = new_prob - original_prob
                
                comparison_data.append({
                    'Player': player_name,
                    'Original': f"{original_prob:.1%}",
                    'What-if': f"{new_prob:.1%}",
                    'Change': f"{change:+.1%}",
                    'Impact': "📈" if change > 0.001 else "📉" if change < -0.001 else "➡️"
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 結果表示
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # 主要な変化をハイライト
        changes = [float(x.replace('%', '').replace('+', '')) for x in comparison_df['Change']]
        max_idx = changes.index(max(changes))
        min_idx = changes.index(min(changes))
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.success(f"📈 **Biggest Winner**: {comparison_df.iloc[max_idx]['Player']} ({comparison_df.iloc[max_idx]['Change']})")
        with col_res2:
            st.error(f"📉 **Biggest Loser**: {comparison_df.iloc[min_idx]['Player']} ({comparison_df.iloc[min_idx]['Change']})")


if __name__ == "__main__":
    main()