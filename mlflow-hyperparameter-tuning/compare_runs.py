"""
Compare and Analyze MLflow Runs

Konsep yang dipelajari:
1. Searching and filtering runs
2. Comparing metrics across runs
3. Finding best models berdasarkan criteria
4. Visualization dengan matplotlib/seaborn
5. MLflow Search API
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set tracking URI
mlflow.set_tracking_uri("file:./mlruns")


def search_runs_example():
    """
    Example 1: Search runs dengan berbagai filters
    """
    print("\n" + "="*60)
    print("🔍 SEARCH RUNS")
    print("="*60)
    
    # Get experiment
    experiment = mlflow.get_experiment_by_name("hyperparameter-tuning-gridsearch")
    
    if experiment is None:
        print("❌ Experiment not found. Run grid_search.py first!")
        return None
    
    # Search all runs dalam experiment
    all_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=10
    )
    
    if len(all_runs) == 0:
        print("No runs found. Run grid_search.py first!")
        return None
    
    print(f"✅ Found {len(all_runs)} runs")
    print("\nTop 5 Runs by Accuracy:")
    print(all_runs[['run_id', 'metrics.accuracy', 'start_time']].head())
    
    return all_runs


def compare_experiments():
    """
    Example 2: Compare runs across different experiments
    """
    print("\n" + "="*60)
    print("📊 COMPARE EXPERIMENTS")
    print("="*60)
    
    experiments = ["hyperparameter-tuning-gridsearch", 
                   "hyperparameter-tuning-randomsearch",
                   "hyperparameter-tuning-optuna"]
    
    all_data = []
    
    for exp_name in experiments:
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp:
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="tags.mlflow.parentRunId != ''",  # Only child runs
                max_results=1000
            )
            if len(runs) > 0:
                runs['experiment'] = exp_name.split('-')[-1]  # Extract method name
                all_data.append(runs)
                print(f"  ✅ {exp_name}: {len(runs)} runs")
    
    if not all_data:
        print("❌ No runs found. Run experiments first!")
        return None
    
    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    
    # Summary statistics
    print("\n📈 Summary Statistics by Method:")
    summary = df.groupby('experiment')['metrics.accuracy'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).round(4)
    print(summary)
    
    return df


def find_best_models():
    """
    Example 3: Find best models berdasarkan different criteria
    """
    print("\n" + "="*60)
    print("🏆 FIND BEST MODELS")
    print("="*60)
    
    client = MlflowClient()
    
    # Get all experiments
    experiments = client.search_experiments()
    
    best_models = []
    
    for exp in experiments:
        if 'hyperparameter-tuning' in exp.name:
            # Search best run dalam experiment ini
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="tags.mlflow.parentRunId != ''",  # Child runs only
                order_by=["metrics.accuracy DESC"],
                max_results=1
            )
            
            if len(runs) > 0:
                best_run = runs.iloc[0]
                best_models.append({
                    'experiment': exp.name,
                    'run_id': best_run['run_id'],
                    'accuracy': best_run['metrics.accuracy'],
                    'params': {k.replace('params.', ''): v 
                              for k, v in best_run.items() 
                              if k.startswith('params.')}
                })
    
    if not best_models:
        print("No models found!")
        return
    
    # Print best models
    print("\n🥇 Best Model per Experiment:")
    for model in sorted(best_models, key=lambda x: x['accuracy'], reverse=True):
        print(f"\n{model['experiment']}:")
        print(f"  Accuracy: {model['accuracy']:.4f}")
        print(f"  Run ID: {model['run_id'][:8]}...")
        print(f"  Top params: {dict(list(model['params'].items())[:3])}")


def visualize_comparison(df):
    """
    Example 4: Visualize comparison dengan matplotlib
    """
    print("\n" + "="*60)
    print("📉 VISUALIZATION")
    print("="*60)
    
    if df is None or len(df) == 0:
        print("No data to visualize!")
        return
    
    # Create figure dengan multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MLflow Hyperparameter Tuning Comparison', fontsize=16, fontweight='bold')
    
    # 1. Boxplot - Accuracy distribution per method
    ax1 = axes[0, 0]
    df.boxplot(column='metrics.accuracy', by='experiment', ax=ax1)
    ax1.set_title('Accuracy Distribution by Method')
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Accuracy')
    plt.sca(ax1)
    plt.xticks(rotation=45)
    
    # 2. Violin plot
    ax2 = axes[0, 1]
    sns.violinplot(data=df, x='experiment', y='metrics.accuracy', ax=ax2)
    ax2.set_title('Accuracy Distribution (Violin Plot)')
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Accuracy')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Histogram - Overall accuracy distribution
    ax3 = axes[1, 0]
    for method in df['experiment'].unique():
        method_data = df[df['experiment'] == method]['metrics.accuracy']
        ax3.hist(method_data, alpha=0.5, label=method, bins=20)
    ax3.set_title('Accuracy Histogram')
    ax3.set_xlabel('Accuracy')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # 4. Bar plot - Mean accuracy dengan error bars
    ax4 = axes[1, 1]
    summary = df.groupby('experiment')['metrics.accuracy'].agg(['mean', 'std'])
    summary.plot(kind='bar', y='mean', yerr='std', ax=ax4, legend=False, capsize=4)
    ax4.set_title('Mean Accuracy by Method (with Std Dev)')
    ax4.set_xlabel('Method')
    ax4.set_ylabel('Mean Accuracy')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'comparison_plots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plots saved to: {output_file}")
    
    # Show plot
    plt.show()


def analyze_parameter_importance(experiment_name="hyperparameter-tuning-gridsearch"):
    """
    Example 5: Analyze parameter importance
    """
    print("\n" + "="*60)
    print("🎯 PARAMETER IMPORTANCE ANALYSIS")
    print("="*60)
    
    exp = mlflow.get_experiment_by_name(experiment_name)
    if not exp:
        print(f"Experiment {experiment_name} not found!")
        return
    
    # Get runs
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.mlflow.parentRunId != ''",
        max_results=1000
    )
    
    if len(runs) == 0:
        print("No runs found!")
        return
    
    # Extract parameter columns
    param_cols = [col for col in runs.columns if col.startswith('params.')]
    
    print(f"\n📊 Analyzing {len(runs)} runs from {experiment_name}")
    print(f"Parameters: {[col.replace('params.', '') for col in param_cols]}")
    
    # Calculate correlation dengan accuracy
    print("\n🔗 Parameter Correlation with Accuracy:")
    
    for param_col in param_cols:
        # Convert to numeric (skip None values)
        param_name = param_col.replace('params.', '')
        try:
            numeric_vals = pd.to_numeric(runs[param_col], errors='coerce')
            if numeric_vals.notna().sum() > 0:
                correlation = numeric_vals.corr(runs['metrics.accuracy'])
                print(f"  {param_name:20s}: {correlation:7.4f}")
        except:
            pass
    
    # Top performing parameter combinations
    print("\n🥇 Top 5 Parameter Combinations:")
    top_runs = runs.nlargest(5, 'metrics.accuracy')
    
    for idx, (_, run) in enumerate(top_runs.iterrows(), 1):
        print(f"\n  Rank {idx}: Accuracy = {run['metrics.accuracy']:.4f}")
        for col in param_cols:
            param_name = col.replace('params.', '')
            print(f"    {param_name}: {run[col]}")


def export_results():
    """
    Example 6: Export results ke CSV untuk further analysis
    """
    print("\n" + "="*60)
    print("💾 EXPORT RESULTS")
    print("="*60)
    
    experiments = ["hyperparameter-tuning-gridsearch", 
                   "hyperparameter-tuning-randomsearch",
                   "hyperparameter-tuning-optuna"]
    
    for exp_name in experiments:
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp:
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=1000
            )
            
            if len(runs) > 0:
                output_file = f"{exp_name}_results.csv"
                runs.to_csv(output_file, index=False)
                print(f"  ✅ Exported {len(runs)} runs to {output_file}")


def main():
    """
    Run all analysis
    """
    print("\n" + "="*70)
    print(" 🔬 MLFLOW RUNS COMPARISON & ANALYSIS")
    print("="*70)
    
    # 1. Search runs
    runs = search_runs_example()
    
    # 2. Compare experiments
    df = compare_experiments()
    
    # 3. Find best models
    find_best_models()
    
    # 4. Visualize (if data available)
    if df is not None and len(df) > 0:
        try:
            visualize_comparison(df)
        except Exception as e:
            print(f"⚠️  Visualization error: {e}")
    
    # 5. Parameter importance
    analyze_parameter_importance()
    
    # 6. Export results
    export_results()
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETED!")
    print("="*70)
    
    print("\n💡 Tips:")
    print("  - Use MLflow UI untuk interactive comparison")
    print("  - Filter runs menggunakan filter_string")
    print("  - Export ke CSV untuk custom analysis")
    print("  - Compare metrics across different experiments")
    
    print("\n📊 Open MLflow UI:")
    print("  mlflow ui --backend-store-uri file:./mlruns")


if __name__ == "__main__":
    main()
