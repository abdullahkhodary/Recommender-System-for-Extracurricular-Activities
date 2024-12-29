from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import matplotlib.pyplot as plt
import seaborn as sns

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
}

output_dir = '/opt/airflow/dags/output'
visualization_dir = '/opt/airflow/dags/visualizations'

def create_output_dir():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)

def extract_students_data_from_postgres(**kwargs):
    pg_hook = PostgresHook(postgres_conn_id='postgres_default', schema='recommender_db')
    df_students = pd.read_sql_query("SELECT * FROM students", pg_hook.get_conn())
    df_students.to_csv(os.path.join(output_dir, 'students.csv'), index=False)
    kwargs['ti'].xcom_push(key='students_data', value=df_students.to_json())

def extract_activities_data_from_postgres(**kwargs):
    pg_hook = PostgresHook(postgres_conn_id='postgres_default', schema='recommender_db')
    df_activities = pd.read_sql_query("SELECT * FROM activities", pg_hook.get_conn())
    df_activities.to_csv(os.path.join(output_dir, 'activities.csv'), index=False)
    kwargs['ti'].xcom_push(key='activities_data', value=df_activities.to_json())

def engineer_students_data(**kwargs):
    ti = kwargs['ti']
    df_students_json = ti.xcom_pull(task_ids='extract_students_data_task', key='students_data')
    df_students = pd.read_json(df_students_json)
    
    # Data engineering steps
    df_students['academicinterest'] = df_students['academicinterest'].str.strip()
    df_students['skills'] = df_students['skills'].apply(lambda x: ', '.join(sorted(x.split(', '))))
    df_students['languages'] = df_students['languages'].apply(lambda x: ', '.join(sorted(x.split(', '))))
    df_students['clubmemberships'] = df_students['clubmemberships'].apply(lambda x: ', '.join(sorted(x.split(', '))))
    
    df_students.to_csv(os.path.join(output_dir, 'students_engineered.csv'), index=False)
    ti.xcom_push(key='students_data_engineered', value=df_students.to_json())

def perform_recommendations(**kwargs):
    ti = kwargs['ti']
    
    # Load the engineered students data and activities data from XCom
    df_students_json = ti.xcom_pull(task_ids='engineer_students_data_task', key='students_data_engineered')
    df_activities_json = ti.xcom_pull(task_ids='extract_activities_data_task', key='activities_data')
    
    df_students = pd.read_json(df_students_json)
    df_activities = pd.read_json(df_activities_json)
    
    # Prepare data for recommendations
    student_features = df_students[['academicinterest', 'extracurricularactivities', 'skills', 'researchinterests']].fillna('')
    activity_features = df_activities[['name', 'description', 'category']].fillna('')
    
    tfidf = TfidfVectorizer(stop_words='english')
    activity_matrix = tfidf.fit_transform(activity_features.apply(lambda x: ' '.join(map(str, x)), axis=1))
    
    df_recommendations_content_based = pd.DataFrame()
    num_recommendations = 5
    
    for student_idx, student_row in df_students.iterrows():
        student_id = student_row['studentid']
        student_matrix = tfidf.transform(student_features.iloc[[student_idx]].apply(lambda x: ' '.join(map(str, x)), axis=1))
        similarity_scores = cosine_similarity(student_matrix, activity_matrix).flatten()
        activity_indices = similarity_scores.argsort()[-num_recommendations:][::-1]
        recommended_activities = df_activities.iloc[activity_indices].copy()
        recommended_activities['studentid'] = student_id
        df_recommendations_content_based = pd.concat([df_recommendations_content_based, recommended_activities], ignore_index=True)
    
    # Collaborative Filtering Recommendation
    student_features = df_students[['extracurricularactivities']].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    student_matrix = tfidf.fit_transform(student_features.apply(lambda x: ' '.join(map(str, x)), axis=1))
    similarity_matrix = cosine_similarity(student_matrix)
    
    recommendations_collaborative_filtering = []
    num_recommendations = 3
    
    for student_idx, student_row in df_students.iterrows():
        student_id = student_row['studentid']
        similarity_scores = similarity_matrix[student_idx]
        similar_student_indices = similarity_scores.argsort()[::-1][1:]
        recommended_activities = []
        activities_count = 0
        
        for idx in similar_student_indices:
            similar_student_activities = df_students.loc[idx, 'extracurricularactivities'].split(', ')
            for activity in similar_student_activities:
                if activity not in student_row['extracurricularactivities'] and activity not in recommended_activities:
                    recommended_activities.append(activity)
                    activities_count += 1
                    if activities_count >= num_recommendations:
                        break
            if activities_count >= num_recommendations:
                break
        
        student_recommendations = pd.DataFrame({
            'studentid': student_id,
            'Recommended_Activity': recommended_activities[:num_recommendations]
        })
        
        recommendations_collaborative_filtering.append(student_recommendations)
    
    df_recommendations_collaborative_filtering = pd.concat(recommendations_collaborative_filtering, ignore_index=True)
    
    # Push the recommendations to XCom
    ti.xcom_push(key='content_based_recommendations', value=df_recommendations_content_based.to_json())
    ti.xcom_push(key='collaborative_filtering_recommendations', value=df_recommendations_collaborative_filtering.to_json())

def load_recommendations(**kwargs):
    ti = kwargs['ti']
    df_recommendations_content_based_json = ti.xcom_pull(task_ids='perform_recommendations_task', key='content_based_recommendations')
    df_recommendations_collaborative_filtering_json = ti.xcom_pull(task_ids='perform_recommendations_task', key='collaborative_filtering_recommendations')
    
    df_recommendations_content_based = pd.read_json(df_recommendations_content_based_json)
    df_recommendations_collaborative_filtering = pd.read_json(df_recommendations_collaborative_filtering_json)
    
    pg_hook = PostgresHook(postgres_conn_id='postgres_default', schema='recommender_db')
    connection = pg_hook.get_conn()
    cursor = connection.cursor()
    
    try:
        cursor.execute("TRUNCATE TABLE recommendations;")
        
        for _, row in df_recommendations_content_based.iterrows():
            student_id = row['studentid']
            recommendation_description = f"{row['name']}"
            activity_id = row.get('activity_id', None)
            provider_id = row.get('provider_id', None)
            cursor.execute(
                "INSERT INTO recommendations (student_id, recommendation_description, activity_id, provider_id) VALUES (%s, %s, %s, %s);",
                (student_id, recommendation_description, activity_id, provider_id)
            )
        
        for _, row in df_recommendations_collaborative_filtering.iterrows():
            student_id = row['studentid']
            recommended_activity = row['Recommended_Activity']
            cursor.execute(
                "INSERT INTO recommendations (student_id, recommendation_description) VALUES (%s, %s);",
                (student_id, recommended_activity)
            )
        
        connection.commit()
    finally:
        cursor.close()
        connection.close()

def visualize_recommendations(**kwargs):
    ti = kwargs['ti']
    df_recommendations_content_based_json = ti.xcom_pull(task_ids='perform_recommendations_task', key='content_based_recommendations')
    df_recommendations_collaborative_filtering_json = ti.xcom_pull(task_ids='perform_recommendations_task', key='collaborative_filtering_recommendations')
    
    df_recommendations_content_based = pd.read_json(df_recommendations_content_based_json)
    df_recommendations_collaborative_filtering = pd.read_json(df_recommendations_collaborative_filtering_json)
    
    # Aggregate recommendations for content-based filtering
    content_based_counts = df_recommendations_content_based['name'].value_counts().reset_index()
    content_based_counts.columns = ['Activity', 'StudentCount']
    
    # Aggregate recommendations for collaborative filtering
    collaborative_counts = df_recommendations_collaborative_filtering['Recommended_Activity'].value_counts().reset_index()
    collaborative_counts.columns = ['Activity', 'StudentCount']
    
    # Plot content-based recommendations
    plt.figure(figsize=(14, 7))
    sns.barplot(data=content_based_counts, x='Activity', y='StudentCount', palette='viridis')
    plt.xticks(rotation=90)
    plt.title('Number of Students Recommended per Activity (Content-Based)')
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, 'content_based_recommendations_counts.png'))
    plt.close()
    
    # Plot collaborative filtering recommendations
    plt.figure(figsize=(14, 7))
    sns.barplot(data=collaborative_counts, x='Activity', y='StudentCount', palette='viridis')
    plt.xticks(rotation=90)
    plt.title('Number of Students Recommended per Activity (Collaborative Filtering)')
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, 'collaborative_filtering_recommendations_counts.png'))
    plt.close()

with DAG(
    'student_recommendation_system',
    default_args=default_args,
    description='A DAG for generating student recommendations based on content-based and collaborative filtering methods.',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
) as dag:

    create_output_task = PythonOperator(
        task_id='create_output_dir_task',
        python_callable=create_output_dir,
    )
    
    extract_students_data_task = PythonOperator(
        task_id='extract_students_data_task',
        python_callable=extract_students_data_from_postgres,
        provide_context=True,
    )

    extract_activities_data_task = PythonOperator(
        task_id='extract_activities_data_task',
        python_callable=extract_activities_data_from_postgres,
        provide_context=True,
    )
    
    engineer_students_data_task = PythonOperator(
        task_id='engineer_students_data_task',
        python_callable=engineer_students_data,
        provide_context=True,
    )

    perform_recommendations_task = PythonOperator(
        task_id='perform_recommendations_task',
        python_callable=perform_recommendations,
        provide_context=True,
    )

    load_recommendations_task = PythonOperator(
        task_id='load_recommendations_task',
        python_callable=load_recommendations,
        provide_context=True,
    )
    
    visualize_recommendations_task = PythonOperator(
        task_id='visualize_recommendations_task',
        python_callable=visualize_recommendations,
        provide_context=True,
    )
    
    create_output_task >> [extract_students_data_task, extract_activities_data_task]
    extract_students_data_task >> engineer_students_data_task
    extract_activities_data_task >> perform_recommendations_task
    engineer_students_data_task >> perform_recommendations_task
    perform_recommendations_task >> [load_recommendations_task, visualize_recommendations_task]
