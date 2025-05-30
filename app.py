import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

# Try to import plotly, fall back to matplotlib if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Using matplotlib for all visualizations.")

# Page config
st.set_page_config(
    page_title="Learning Engagement Tracker",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Download functionality for plots
def create_download_link(fig, filename):
    """Create a download button for matplotlib figures"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    return buffer.getvalue()

# Generate sample data
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    
    # Student data
    students = [f"Student_{i:03d}" for i in range(1, 501)]
    courses = ["Python Programming", "Data Science", "Web Development", "Machine Learning", "Database Systems"]
    
    data = []
    start_date = datetime.now() - timedelta(days=90)
    
    for student in students:
        for course in courses:
            if np.random.random() > 0.3:  # 70% enrollment rate
                hours_spent = np.random.exponential(15) + 5  # 5-50+ hours
                performance = min(100, max(0, 
                    60 + (hours_spent * 1.2) + np.random.normal(0, 15)))
                
                # Weekly activity over 12 weeks
                weekly_hours = []
                for week in range(12):
                    base_hours = hours_spent / 12
                    variation = np.random.normal(1, 0.3)
                    week_hours = max(0, base_hours * variation)
                    weekly_hours.append(week_hours)
                
                data.append({
                    'student_id': student,
                    'course': course,
                    'total_hours': hours_spent,
                    'performance_score': performance,
                    'weekly_hours': weekly_hours,
                    'engagement_level': 'High' if hours_spent > 25 else 'Medium' if hours_spent > 15 else 'Low',
                    'completion_rate': min(100, (hours_spent / 30) * 100)
                })
    
    return pd.DataFrame(data)

# Save plot to GCP
def save_plot_locally(fig, filename):
    """Save plot and provide download option"""
    buffer = create_download_link(fig, filename)
    st.download_button(
        label=f"📥 Download {filename}",
        data=buffer,
        file_name=filename,
        mime="image/png",
        help="Click to download the plot as PNG"
    )

# Main app
def main():
    # Initialize app
    st.markdown('<h1 class="main-header">📚 Online Learning Engagement Tracker</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    df = generate_sample_data()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    selected_courses = st.sidebar.multiselect(
        "Select Courses", 
        options=df['course'].unique(), 
        default=df['course'].unique()
    )
    
    engagement_filter = st.sidebar.selectbox(
        "Engagement Level", 
        options=['All'] + list(df['engagement_level'].unique())
    )
    
    # Filter data
    filtered_df = df[df['course'].isin(selected_courses)]
    if engagement_filter != 'All':
        filtered_df = filtered_df[filtered_df['engagement_level'] == engagement_filter]
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_students = len(filtered_df['student_id'].unique())
        st.metric("Total Students", total_students)
    
    with col2:
        avg_hours = filtered_df['total_hours'].mean()
        st.metric("Avg Hours/Course", f"{avg_hours:.1f}")
    
    with col3:
        avg_performance = filtered_df['performance_score'].mean()
        st.metric("Avg Performance", f"{avg_performance:.1f}%")
    
    with col4:
        completion_rate = filtered_df['completion_rate'].mean()
        st.metric("Avg Completion", f"{completion_rate:.1f}%")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔥 Heatmap", "📈 Trends", "🎯 Insights"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Engagement per course bar chart
            course_engagement = filtered_df.groupby('course')['total_hours'].mean().sort_values(ascending=True)
            
            if PLOTLY_AVAILABLE:
                fig_bar = px.bar(
                    x=course_engagement.values,
                    y=course_engagement.index,
                    orientation='h',
                    title="Average Hours per Course",
                    color=course_engagement.values,
                    color_continuous_scale='viridis'
                )
                fig_bar.update_layout(height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                bars = ax.barh(course_engagement.index, course_engagement.values, color='viridis')
                ax.set_title('Average Hours per Course', fontsize=14)
                ax.set_xlabel('Hours')
                st.pyplot(fig)
        
        with col2:
            # Performance distribution
            if PLOTLY_AVAILABLE:
                fig_hist = px.histogram(
                    filtered_df, 
                    x='performance_score', 
                    nbins=20,
                    title="Performance Score Distribution",
                    color_discrete_sequence=['#ff7f0e']
                )
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(filtered_df['performance_score'], bins=20, color='#ff7f0e', alpha=0.7)
                ax.set_title('Performance Score Distribution', fontsize=14)
                ax.set_xlabel('Performance Score')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
    
    with tab2:
        st.subheader("🔥 Hours vs Performance Heatmap")
        
        # Create heatmap data
        filtered_df['hours_bin'] = pd.cut(filtered_df['total_hours'], bins=10)
        filtered_df['perf_bin'] = pd.cut(filtered_df['performance_score'], bins=10)
        
        heatmap_data = filtered_df.groupby(['hours_bin', 'perf_bin']).size().unstack(fill_value=0)
        
        # Matplotlib heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
        ax.set_title('Student Distribution: Hours Spent vs Performance Score', fontsize=16)
        ax.set_xlabel('Performance Score Bins', fontsize=12)
        ax.set_ylabel('Hours Spent Bins', fontsize=12)
        
        st.pyplot(fig)
        
        # Download option
        col1, col2 = st.columns([1, 4])
        with col1:
            save_plot_locally(fig, 'heatmap_hours_vs_performance.png')
    
    with tab3:
        st.subheader("📈 Weekly Activity Trends")
        
        # Prepare weekly data
        weekly_data = []
        for _, row in filtered_df.iterrows():
            for week, hours in enumerate(row['weekly_hours'], 1):
                weekly_data.append({
                    'Week': week,
                    'Hours': hours,
                    'Course': row['course'],
                    'Student': row['student_id']
                })
        
        weekly_df = pd.DataFrame(weekly_data)
        weekly_summary = weekly_df.groupby(['Week', 'Course'])['Hours'].mean().reset_index()
        
        # Line chart
        if PLOTLY_AVAILABLE:
            fig_line = px.line(
                weekly_summary, 
                x='Week', 
                y='Hours', 
                color='Course',
                title='Weekly Learning Activity Trends',
                markers=True
            )
            fig_line.update_layout(height=500)
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            for course in weekly_summary['Course'].unique():
                course_data = weekly_summary[weekly_summary['Course'] == course]
                ax.plot(course_data['Week'], course_data['Hours'], marker='o', label=course)
            ax.set_title('Weekly Learning Activity Trends', fontsize=14)
            ax.set_xlabel('Week')
            ax.set_ylabel('Hours')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        # Overall trend
        overall_weekly = weekly_df.groupby('Week')['Hours'].mean().reset_index()
        if PLOTLY_AVAILABLE:
            fig_overall = px.area(
                overall_weekly, 
                x='Week', 
                y='Hours',
                title='Overall Weekly Activity Trend',
                color_discrete_sequence=['#2ca02c']
            )
            st.plotly_chart(fig_overall, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.fill_between(overall_weekly['Week'], overall_weekly['Hours'], alpha=0.6, color='#2ca02c')
            ax.plot(overall_weekly['Week'], overall_weekly['Hours'], color='#2ca02c', linewidth=2)
            ax.set_title('Overall Weekly Activity Trend', fontsize=14)
            ax.set_xlabel('Week')
            ax.set_ylabel('Hours')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
    
    with tab4:
        st.subheader("🎯 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top performers
            st.write("**🏆 Top Performing Courses**")
            top_courses = filtered_df.groupby('course')['performance_score'].mean().sort_values(ascending=False)
            for i, (course, score) in enumerate(top_courses.head(3).items(), 1):
                st.write(f"{i}. {course}: {score:.1f}%")
            
            # Engagement insights
            st.write("**⚡ Engagement Insights**")
            high_engagement = len(filtered_df[filtered_df['engagement_level'] == 'High'])
            total = len(filtered_df)
            st.write(f"• {(high_engagement/total)*100:.1f}% high engagement rate")
            
            correlation = filtered_df['total_hours'].corr(filtered_df['performance_score'])
            st.write(f"• Hours-Performance correlation: {correlation:.2f}")
        
        with col2:
            # Course comparison radar
            course_metrics = filtered_df.groupby('course').agg({
                'total_hours': 'mean',
                'performance_score': 'mean',
                'completion_rate': 'mean'
            }).round(2)
            
            if PLOTLY_AVAILABLE:
                fig_radar = go.Figure()
                
                for course in course_metrics.index[:3]:  # Top 3 courses
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[
                            course_metrics.loc[course, 'total_hours']/course_metrics['total_hours'].max()*100,
                            course_metrics.loc[course, 'performance_score'],
                            course_metrics.loc[course, 'completion_rate']
                        ],
                        theta=['Hours', 'Performance', 'Completion'],
                        fill='toself',
                        name=course
                    ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=True,
                    title="Course Comparison (Radar Chart)"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                # Matplotlib alternative - simple bar chart comparison
                fig, ax = plt.subplots(figsize=(10, 6))
                x = range(len(course_metrics.index[:3]))
                width = 0.25
                
                hours_norm = course_metrics['total_hours'][:3] / course_metrics['total_hours'].max() * 100
                performance = course_metrics['performance_score'][:3]
                completion = course_metrics['completion_rate'][:3]
                
                ax.bar([i - width for i in x], hours_norm, width, label='Hours (normalized)', alpha=0.8)
                ax.bar(x, performance, width, label='Performance', alpha=0.8)
                ax.bar([i + width for i in x], completion, width, label='Completion', alpha=0.8)
                
                ax.set_xlabel('Courses')
                ax.set_ylabel('Scores')
                ax.set_title('Course Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(course_metrics.index[:3], rotation=45)
                ax.legend()
                st.pyplot(fig)
        
        # Detailed data table
        st.subheader("📋 Detailed Data")
        display_df = filtered_df[['student_id', 'course', 'total_hours', 'performance_score', 'engagement_level']].copy()
        display_df.columns = ['Student ID', 'Course', 'Hours', 'Performance (%)', 'Engagement']
        st.dataframe(display_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**📊 Online Learning Engagement Tracker** | "
        "Built with Streamlit | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

if __name__ == "__main__":
    main()
