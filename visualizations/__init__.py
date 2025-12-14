"""Visualization utilities for MedVLM-Probe"""

from typing import Optional
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_gauge(score: float, title: str = "MedVLM-Probe Score") -> Optional["go.Figure"]:
    """Create a gauge chart for overall score"""
    if not PLOTLY_AVAILABLE:
        print(" Plotly not installed. Run: pip install plotly")
        return None
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 70, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': '#ff6b6b'},
                {'range': [50, 70], 'color': '#ffd93d'},
                {'range': [70, 100], 'color': '#6bcb77'}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 70}
        }
    ))
    fig.update_layout(height=400)
    return fig


def plot_scores_by_type(scores_by_type: dict, title: str = "Scores by Test Type") -> Optional["go.Figure"]:
    """Create bar chart of scores by test type"""
    if not PLOTLY_AVAILABLE:
        print(" Plotly not installed. Run: pip install plotly")
        return None
    
    df = pd.DataFrame([
        {"Test Type": k.replace('_', ' ').title(), "Score": v}
        for k, v in scores_by_type.items()
    ])
    
    fig = px.bar(
        df, x='Test Type', y='Score',
        color='Score',
        color_continuous_scale=['#ff6b6b', '#ffd93d', '#6bcb77'],
        range_color=[0, 100],
        title=title
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Target: 70%")
    fig.update_layout(height=450)
    return fig


def plot_confusion_matrix(results_df: pd.DataFrame, title: str = "Classification Confusion Matrix") -> Optional["go.Figure"]:
    """Create confusion matrix heatmap"""
    if not PLOTLY_AVAILABLE:
        print(" Plotly not installed. Run: pip install plotly")
        return None
    
    class_df = results_df[results_df['test_type'] == 'classification']
    if len(class_df) == 0:
        return None
    
    confusion = pd.crosstab(class_df['true_label'], class_df['predicted'])
    
    fig = px.imshow(
        confusion,
        labels=dict(x="Predicted", y="True Label", color="Count"),
        title=title,
        color_continuous_scale='Blues',
        text_auto=True
    )
    fig.update_layout(height=400)
    return fig


def plot_results_summary(results_df: pd.DataFrame, title: str = "Test Results Summary") -> Optional["go.Figure"]:
    """Create stacked bar chart of pass/fail by test type"""
    if not PLOTLY_AVAILABLE:
        print(" Plotly not installed. Run: pip install plotly")
        return None
    
    summary = results_df.groupby('test_type')['passed'].agg(['sum', 'count']).reset_index()
    summary['failed'] = summary['count'] - summary['sum']
    summary.columns = ['Test Type', 'Passed', 'Total', 'Failed']
    summary['Test Type'] = summary['Test Type'].str.replace('_', ' ').str.title()
    
    fig = go.Figure(data=[
        go.Bar(name='Passed', x=summary['Test Type'], y=summary['Passed'], marker_color='#6bcb77'),
        go.Bar(name='Failed', x=summary['Test Type'], y=summary['Failed'], marker_color='#ff6b6b')
    ])
    fig.update_layout(barmode='stack', title=title, height=400)
    return fig


class ProbeVisualizer:
    """Visualizer for probe results"""
    
    def __init__(self, results: "ProbeResults"):
        self.results = results
        self.df = results.to_dataframe()
        self.scores = results.scores
        
    def plot_all(self, show: bool = True) -> dict:
        """Generate all visualizations"""
        figs = {
            "gauge": plot_gauge(self.scores.overall),
            "by_type": plot_scores_by_type(self.scores.by_type),
            "confusion": plot_confusion_matrix(self.df),
            "summary": plot_results_summary(self.df)
        }
        
        if show and PLOTLY_AVAILABLE:
            for name, fig in figs.items():
                if fig is not None:
                    fig.show()
        
        return figs
    
    def save_html(self, output_dir: str = "./results"):
        """Save all visualizations as HTML"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        figs = self.plot_all(show=False)
        
        for name, fig in figs.items():
            if fig is not None:
                path = os.path.join(output_dir, f"plot_{name}.html")
                fig.write_html(path)
                print(f"   Saved: {path}")
