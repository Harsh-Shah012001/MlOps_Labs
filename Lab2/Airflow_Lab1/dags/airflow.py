from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow import configuration as conf
from src.lab import load_wine_data, preprocess_wine_data, train_wine_clusters, evaluate_wine_clusters
# Enable pickle for XCom
conf.set('core', 'enable_xcom_pickling', 'True')

default_args = {
    'owner': 'harsh_shah',
    'start_date': datetime(2025, 1, 15),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'wine_clustering_pipeline',
    default_args=default_args,
    description='Airflow DAG for clustering wine dataset',
    schedule_interval=None,
    catchup=False,
)

# Define tasks
load_wine_data_task = PythonOperator(
    task_id='load_wine_data',
    python_callable=load_wine_data,
    dag=dag,
)

preprocess_wine_data_task = PythonOperator(
    task_id='preprocess_wine_data',
    python_callable=preprocess_wine_data,
    op_args=[load_wine_data_task.output],
    dag=dag,
)

train_wine_clusters_task = PythonOperator(
    task_id='train_wine_clusters',
    python_callable=train_wine_clusters,
    op_args=[preprocess_wine_data_task.output, "wine_clusters.sav"],
    dag=dag,
)

evaluate_wine_clusters_task = PythonOperator(
    task_id='evaluate_wine_clusters',
    python_callable=evaluate_wine_clusters,
    op_args=["wine_clusters.sav", train_wine_clusters_task.output],
    dag=dag,
)

# Set dependencies
load_wine_data_task >> preprocess_wine_data_task >> train_wine_clusters_task >> evaluate_wine_clusters_task

if __name__ == "__main__":
    dag.cli()