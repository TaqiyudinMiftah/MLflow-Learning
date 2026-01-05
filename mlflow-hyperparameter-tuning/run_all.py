"""
Quick Start Script - Run all experiments sequentially

Script ini akan menjalankan semua experiments secara berurutan untuk demo
"""

import subprocess
import sys
import time


def run_script(script_name, description):
    """Run a Python script dan print status"""
    print("\n" + "="*70)
    print(f"🚀 Running: {description}")
    print("="*70)
    
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        elapsed = time.time() - start_time
        print(f"\n✅ Completed in {elapsed:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def main():
    print("\n" + "="*70)
    print(" 🎯 MLFLOW HYPERPARAMETER TUNING - QUICK START")
    print("="*70)
    print("\nThis will run all experiments sequentially:")
    print("  1. Grid Search (slower, exhaustive)")
    print("  2. Random Search (faster, random sampling)")
    print("  3. Optuna Bayesian Optimization (intelligent search)")
    print("  4. Autologging Demo (automatic logging)")
    print("  5. Compare & Analyze Results")
    print("\n" + "="*70)
    
    input("\nPress ENTER to start...")
    
    scripts = [
        ("grid_search.py", "Grid Search Hyperparameter Tuning"),
        ("random_search.py", "Random Search Hyperparameter Tuning"),
        ("optuna_tuning.py", "Optuna Bayesian Optimization"),
        ("autologging_demo.py", "MLflow Autologging Demo"),
        ("compare_runs.py", "Compare & Analyze Results"),
    ]
    
    results = {}
    total_start = time.time()
    
    for script, desc in scripts:
        success = run_script(script, desc)
        results[script] = success
        
        if not success:
            print(f"\n⚠️  {script} failed. Continue? (y/n)")
            choice = input().lower()
            if choice != 'y':
                break
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "="*70)
    print(" 📊 SUMMARY")
    print("="*70)
    print(f"\nTotal time: {total_elapsed:.2f} seconds ({total_elapsed/60:.1f} minutes)")
    print("\nResults:")
    for script, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {script:30s} {status}")
    
    # Next steps
    print("\n" + "="*70)
    print(" 🎯 NEXT STEPS")
    print("="*70)
    print("\n1. 📊 View results in MLflow UI:")
    print("     mlflow ui --backend-store-uri file:./mlruns")
    print("     Open: http://localhost:5000")
    
    print("\n2. 🔍 Compare experiments:")
    print("     - Open MLflow UI")
    print("     - Select multiple runs")
    print("     - Click 'Compare'")
    
    print("\n3. 📈 Check generated files:")
    print("     - comparison_plots.png")
    print("     - *_results.csv")
    print("     - optimization_history.html (Optuna)")
    
    print("\n4. 📚 Next learning topics:")
    print("     - MLflow Projects")
    print("     - Custom Models (mlflow.pyfunc)")
    print("     - Model Evaluation")
    print("     - Production Deployment")


if __name__ == "__main__":
    main()
