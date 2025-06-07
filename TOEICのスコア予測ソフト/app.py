import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from toeic_simulation import (
    EloRating, TOEICTestConfig, UserAgent, 
    QuestionAgent, TOEICSimulation
)
import io
import base64

st.set_page_config(
    page_title="TOEIC Score Prediction Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def run_simulation_cached(initial_ratings, num_trials, k_factor):
    """キャッシュされたシミュレーション実行"""
    simulation = TOEICSimulation()
    simulation.question_agent = QuestionAgent()
    
    # K係数を設定
    results = {}
    for rating in initial_ratings:
        user_agent = UserAgent(rating)
        user_agent.elo_system.k_factor = k_factor
        
        trial_results = {
            'scores': [],
            'correct_answers': [],
            'rating_history': [rating]
        }
        
        for _ in range(num_trials):
            correct, score = simulation.run_single_test(user_agent)
            trial_results['correct_answers'].append(correct)
            trial_results['scores'].append(score)
            trial_results['rating_history'].append(user_agent.rating)
        
        results[rating] = trial_results
    
    return results

def create_score_distribution_plot(results, initial_ratings):
    """Plotlyを使用してスコア分布を作成"""
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[f"Initial Rating: {rating}" for rating in initial_ratings],
        specs=[[{"secondary_y": False}]*4]*2
    )
    
    colors = px.colors.qualitative.Set3
    
    for i, rating in enumerate(initial_ratings):
        row = i // 4 + 1
        col = i % 4 + 1
        
        scores = results[rating]['scores']
        
        fig.add_trace(
            go.Histogram(
                x=scores,
                nbinsx=30,
                name=f"Rating {rating}",
                marker_color=colors[i],
                opacity=0.7,
                showlegend=False
            ),
            row=row, col=col
        )
        
        # 平均線を追加
        mean_score = np.mean(scores)
        fig.add_vline(
            x=mean_score,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_score:.1f}",
            row=row, col=col
        )
    
    fig.update_layout(
        title_text="TOEIC Score Distribution",
        height=600,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Score")
    fig.update_yaxes(title_text="Frequency")
    
    return fig

def create_rating_evolution_plot(results, initial_ratings):
    """レーティング変化のプロット"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    for i, rating in enumerate(initial_ratings):
        rating_history = results[rating]['rating_history']
        fig.add_trace(
            go.Scatter(
                x=list(range(len(rating_history))),
                y=rating_history,
                mode='lines',
                name=f'Initial Rating: {rating}',
                line=dict(color=colors[i], width=2),
                opacity=0.8
            )
        )
    
    # 問題レーティングの基準線
    fig.add_hline(
        y=1500,
        line_dash="dash",
        line_color="red",
        annotation_text="Question Rating: 1500"
    )
    
    fig.update_layout(
        title="Rating Evolution During Simulation",
        xaxis_title="Trial Number",
        yaxis_title="Rating",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_statistics_table(results, initial_ratings):
    """統計テーブルの作成"""
    stats_data = []
    
    for rating in initial_ratings:
        scores = results[rating]['scores']
        final_rating = results[rating]['rating_history'][-1]
        last_100_ratings = results[rating]['rating_history'][-100:]
        convergence = np.mean(last_100_ratings)
        
        stats_data.append({
            'Initial Rating': rating,
            'Mean Score': f"{np.mean(scores):.1f}",
            'Std Dev': f"{np.std(scores):.1f}",
            'Min Score': f"{np.min(scores):.0f}",
            'Max Score': f"{np.max(scores):.0f}",
            'Final Rating': f"{final_rating:.1f}",
            'Convergence': f"{convergence:.1f}"
        })
    
    return pd.DataFrame(stats_data)

def main():
    st.title("📊 TOEIC Score Prediction Simulator")
    st.markdown("### Multi-Agent Simulation using Elo Rating System")
    
    # サイドバー設定
    st.sidebar.header("⚙️ Simulation Parameters")
    
    # パラメータ設定
    num_trials = st.sidebar.slider(
        "Number of Trials",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100,
        help="Number of TOEIC tests each agent will take"
    )
    
    k_factor = st.sidebar.slider(
        "K-Factor",
        min_value=8,
        max_value=64,
        value=32,
        step=8,
        help="Controls how much ratings change after each test"
    )
    
    # 初期レーティング選択
    st.sidebar.subheader("Initial Ratings")
    available_ratings = [1200, 1300, 1400, 1500, 1600, 1700, 1800]
    selected_ratings = st.sidebar.multiselect(
        "Select Initial Ratings",
        available_ratings,
        default=available_ratings,
        help="Choose which initial ratings to simulate"
    )
    
    if not selected_ratings:
        st.warning("Please select at least one initial rating.")
        return
    
    # シミュレーション実行ボタン
    if st.sidebar.button("🚀 Run Simulation", type="primary"):
        with st.spinner("Running simulation... This may take a few moments."):
            results = run_simulation_cached(selected_ratings, num_trials, k_factor)
            st.session_state.results = results
            st.session_state.selected_ratings = selected_ratings
    
    # 結果表示
    if 'results' in st.session_state:
        results = st.session_state.results
        selected_ratings = st.session_state.selected_ratings
        
        # タブで結果を整理
        tab1, tab2, tab3 = st.tabs(["📊 Score Distribution", "📈 Rating Evolution", "📋 Statistics"])
        
        with tab1:
            st.subheader("Score Distribution Analysis")
            score_fig = create_score_distribution_plot(results, selected_ratings)
            st.plotly_chart(score_fig, use_container_width=True)
            
            # スコア範囲分析
            st.subheader("Score Range Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # 各レーティングの平均スコア
                avg_scores = [np.mean(results[rating]['scores']) for rating in selected_ratings]
                score_df = pd.DataFrame({
                    'Initial Rating': selected_ratings,
                    'Average Score': avg_scores
                })
                
                fig_bar = px.bar(
                    score_df,
                    x='Initial Rating',
                    y='Average Score',
                    title="Average Scores by Initial Rating",
                    color='Average Score',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # スコア分布の箱ひげ図
                all_scores = []
                all_ratings = []
                for rating in selected_ratings:
                    scores = results[rating]['scores']
                    all_scores.extend(scores)
                    all_ratings.extend([rating] * len(scores))
                
                box_df = pd.DataFrame({
                    'Initial Rating': all_ratings,
                    'Score': all_scores
                })
                
                fig_box = px.box(
                    box_df,
                    x='Initial Rating',
                    y='Score',
                    title="Score Distribution (Box Plot)"
                )
                st.plotly_chart(fig_box, use_container_width=True)
        
        with tab2:
            st.subheader("Rating Evolution Over Time")
            evolution_fig = create_rating_evolution_plot(results, selected_ratings)
            st.plotly_chart(evolution_fig, use_container_width=True)
            
            # 収束分析
            st.subheader("Convergence Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # 最終100試行の標準偏差（収束の安定性）
                stability_data = []
                for rating in selected_ratings:
                    last_100 = results[rating]['rating_history'][-100:]
                    stability = np.std(last_100)
                    stability_data.append({
                        'Initial Rating': rating,
                        'Stability (Std Dev)': stability
                    })
                
                stability_df = pd.DataFrame(stability_data)
                fig_stability = px.bar(
                    stability_df,
                    x='Initial Rating',
                    y='Stability (Std Dev)',
                    title="Rating Stability (Lower is More Stable)"
                )
                st.plotly_chart(fig_stability, use_container_width=True)
            
            with col2:
                # 収束値と初期値の比較
                convergence_data = []
                for rating in selected_ratings:
                    final_rating = np.mean(results[rating]['rating_history'][-100:])
                    convergence_data.append({
                        'Initial Rating': rating,
                        'Final Rating': final_rating,
                        'Difference': final_rating - rating
                    })
                
                conv_df = pd.DataFrame(convergence_data)
                fig_conv = px.scatter(
                    conv_df,
                    x='Initial Rating',
                    y='Final Rating',
                    size=abs(conv_df['Difference']),
                    color='Difference',
                    title="Initial vs Final Rating",
                    color_continuous_scale='RdBu'
                )
                fig_conv.add_trace(
                    go.Scatter(
                        x=[min(selected_ratings), max(selected_ratings)],
                        y=[min(selected_ratings), max(selected_ratings)],
                        mode='lines',
                        name='y=x',
                        line=dict(dash='dash', color='gray')
                    )
                )
                st.plotly_chart(fig_conv, use_container_width=True)
        
        with tab3:
            st.subheader("Simulation Statistics")
            stats_df = create_statistics_table(results, selected_ratings)
            st.dataframe(stats_df, use_container_width=True)
            
            # ダウンロード機能
            st.subheader("Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV ダウンロード
                csv = stats_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Statistics (CSV)",
                    data=csv,
                    file_name="toeic_simulation_stats.csv",
                    mime="text/csv"
                )
            
            with col2:
                # 詳細データのダウンロード
                detailed_data = []
                for rating in selected_ratings:
                    for i, (score, correct) in enumerate(zip(results[rating]['scores'], results[rating]['correct_answers'])):
                        detailed_data.append({
                            'Initial_Rating': rating,
                            'Trial': i + 1,
                            'Score': score,
                            'Correct_Answers': correct,
                            'Rating_After_Test': results[rating]['rating_history'][i + 1]
                        })
                
                detailed_df = pd.DataFrame(detailed_data)
                detailed_csv = detailed_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Detailed Data (CSV)",
                    data=detailed_csv,
                    file_name="toeic_simulation_detailed.csv",
                    mime="text/csv"
                )
    
    # アプリケーション情報
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This simulator uses the Elo Rating System to model
        TOEIC test performance. Each user agent competes
        against questions with a fixed rating of 1500.
        
        **Parameters:**
        - **Trials**: Number of TOEIC tests per agent
        - **K-Factor**: Rating change sensitivity
        - **Initial Ratings**: Starting skill levels
        """)
        
        # TOEIC テスト情報
        st.markdown("### 📝 TOEIC Test Info")
        config = TOEICTestConfig()
        st.markdown(f"""
        - **Total Questions**: {config.total_questions}
        - **Listening**: {config.listening_questions} questions
        - **Reading**: {config.reading_questions} questions
        - **Score Range**: 10-990 points (5-point increments)
        """)

if __name__ == "__main__":
    main()