import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Generate sample data
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    
    # Student data
    students = [f"Student_{i:03d}" for i in range(1, 501)]
    courses = ["Python Programming", "Data Science", "Web Development", "Machine Learning", "Database Systems"]
    
    data = []
    
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

# Create visualizations using Streamlit's built-in charts
def create_bar_chart(data, title):
    st.subheader(title)
    st.bar_chart(data)

def create_line_chart(data, title):
    st.subheader(title)
    st.line_chart(data)

def create_area_chart(data, title):
    st.subheader(title)
    st.area_chart(data)

# Main app
def main():
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
    st.header("📊 Key Performance Indicators")
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
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔥 Analysis", "📈 Trends", "🎯 Insights"])
    
    with tab1:
        st.header("Course Engagement Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Engagement per course
            course_engagement = filtered_df.groupby('course')['total_hours'].mean()
            create_bar_chart(course_engagement, "Average Hours per Course")
        
        with col2:
            # Performance by course
            course_performance = filtered_df.groupby('course')['performance_score'].mean()
            create_bar_chart(course_performance, "Average Performance by Course")
        
        # Engagement level distribution
        st.subheader("Engagement Level Distribution")
        engagement_counts = filtered_df['engagement_level'].value_counts()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("High Engagement", engagement_counts.get('High', 0))
        with col2:
            st.metric("Medium Engagement", engagement_counts.get('Medium', 0))
        with col3:
            st.metric("Low Engagement", engagement_counts.get('Low', 0))
    
    with tab2:
        st.header("🔥 Learning Pattern Analysis")
        
        # Hours vs Performance Analysis
        st.subheader("Hours vs Performance Correlation")
        
        # Create bins for analysis
        filtered_df['hours_bin'] = pd.cut(filtered_df['total_hours'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        filtered_df['perf_bin'] = pd.cut(filtered_df['performance_score'], bins=5, labels=['Poor', 'Below Avg', 'Average', 'Good', 'Excellent'])
        
        # Display correlation matrix
        correlation = filtered_df['total_hours'].corr(filtered_df['performance_score'])
        st.markdown(f"""
        <div class="insight-box">
            <h4>📊 Key Finding</h4>
            <p>Strong correlation between study hours and performance: <strong>{correlation:.3f}</strong></p>
            <p>Students who spend more time studying tend to perform significantly better!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Hours distribution by performance level
        perf_hours = filtered_df.groupby('perf_bin')['total_hours'].mean()
        create_bar_chart(perf_hours, "Average Study Hours by Performance Level")
        
        # Course difficulty analysis (inverse of performance)
        course_difficulty = 100 - filtered_df.groupby('course')['performance_score'].mean()
        create_bar_chart(course_difficulty, "Course Difficulty Index (Higher = More Challenging)")
    
    with tab3:
        st.header("📈 Weekly Activity Trends")
        
        # Prepare weekly data
        weekly_data = []
        for _, row in filtered_df.iterrows():
            for week, hours in enumerate(row['weekly_hours'], 1):
                weekly_data.append({
                    'Week': week,
                    'Hours': hours,
                    'Course': row['course']
                })
        
        weekly_df = pd.DataFrame(weekly_data)
        
        # Overall weekly trend
        overall_weekly = weekly_df.groupby('Week')['Hours'].mean()
        create_line_chart(overall_weekly, "Overall Weekly Activity Trend")
        
        # Course-wise weekly trends
        st.subheader("Weekly Trends by Course")
        for course in selected_courses:
            course_weekly = weekly_df[weekly_df['Course'] == course].groupby('Week')['Hours'].mean()
            st.write(f"**{course}**")
            st.line_chart(course_weekly)
        
        # Weekly engagement patterns
        st.subheader("Weekly Engagement Insights")
        peak_week = overall_weekly.idxmax()
        low_week = overall_weekly.idxmin()
        
        st.markdown(f"""
        <div class="insight-box">
            <h4>📅 Weekly Patterns</h4>
            <p><strong>Peak Activity:</strong> Week {peak_week} ({overall_weekly[peak_week]:.1f} hours average)</p>
            <p><strong>Lowest Activity:</strong> Week {low_week} ({overall_weekly[low_week]:.1f} hours average)</p>
            <p><strong>Trend:</strong> {'Increasing' if overall_weekly.iloc[-1] > overall_weekly.iloc[0] else 'Decreasing'} engagement over time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.header("🎯 Detailed Insights & Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Performers")
            
            # Top performing courses
            top_courses = filtered_df.groupby('course')['performance_score'].mean().sort_values(ascending=False)
            for i, (course, score) in enumerate(top_courses.items(), 1):
                st.write(f"{i}. **{course}**: {score:.1f}%")
            
            st.subheader("⚡ Engagement Statistics")
            high_engagement_pct = (len(filtered_df[filtered_df['engagement_level'] == 'High']) / len(filtered_df)) * 100
            st.write(f"• High engagement rate: **{high_engagement_pct:.1f}%**")
            
            avg_completion = filtered_df['completion_rate'].mean()
            st.write(f"• Average completion rate: **{avg_completion:.1f}%**")
            
            # Study pattern insights
            efficient_learners = filtered_df[
                (filtered_df['total_hours'] < filtered_df['total_hours'].median()) & 
                (filtered_df['performance_score'] > filtered_df['performance_score'].median())
            ]
            st.write(f"• Efficient learners (high performance, low hours): **{len(efficient_learners)}** students")
        
        with col2:
            st.subheader("📈 Performance Distribution")
            
            perf_ranges = {
                'Excellent (90-100%)': len(filtered_df[filtered_df['performance_score'] >= 90]),
                'Good (80-89%)': len(filtered_df[(filtered_df['performance_score'] >= 80) & (filtered_df['performance_score'] < 90)]),
                'Average (70-79%)': len(filtered_df[(filtered_df['performance_score'] >= 70) & (filtered_df['performance_score'] < 80)]),
                'Below Average (<70%)': len(filtered_df[filtered_df['performance_score'] < 70])
            }
            
            for range_name, count in perf_ranges.items():
                percentage = (count / len(filtered_df)) * 100
                st.write(f"• **{range_name}**: {count} students ({percentage:.1f}%)")
            
            st.subheader("🎯 Recommendations")
            st.markdown("""
            <div class="insight-box">
                <h4>💡 Key Recommendations</h4>
                <ul>
                    <li>Students should aim for <strong>20+ hours</strong> per course for optimal performance</li>
                    <li>Focus on <strong>consistent weekly study</strong> rather than cramming</li>
                    <li>Courses with lower performance may need <strong>additional support</strong></li>
                    <li>High-engagement students show <strong>significantly better outcomes</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed data table
        st.subheader("📋 Student Performance Data")
        display_df = filtered_df[['student_id', 'course', 'total_hours', 'performance_score', 'engagement_level', 'completion_rate']].copy()
        display_df.columns = ['Student ID', 'Course', 'Hours', 'Performance (%)', 'Engagement', 'Completion (%)']
        display_df = display_df.round(1)
        
        # Add search functionality
        search_term = st.text_input("🔍 Search students or courses:", "")
        if search_term:
            mask = display_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            display_df = display_df[mask]
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download data as CSV
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"learning_engagement_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"**📊 Online Learning Engagement Tracker** | "
        f"Built with Streamlit | "
        f"Analyzing {len(filtered_df)} student records | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

if __name__ == "__main__":
    main()
