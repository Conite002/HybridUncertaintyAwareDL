import os
import json
import torch
from itertools import product
import pandas as pd
from datetime import datetime
from typing import Dict, List
import wandb

from Models.base_model import UncertaintyAwareModel
from Scripts.preprocess import DataPreprocessor
from Scripts.train import Trainer

def run_experiment(config: Dict) -> Dict:
    """
    Run a single experiment with given configuration
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary containing experiment results
    """
    # Set random seeds
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    
    # Initialize data preprocessor and get data loaders
    preprocessor = DataPreprocessor(config)
    train_loader, val_loader, test_loader = preprocessor.create_data_loaders(
        config['dataset'],
        batch_size=config['batch_size']
    )
    
    # Initialize model
    model = UncertaintyAwareModel(
        model_name=config['model_name'],
        num_classes=2,  # Binary classification for all datasets
        pretrained=True,
        dropout_rate=config['dropout_rate']
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config
    )
    
    # Train model
    trainer.train(num_epochs=config['num_epochs'])
    
    return trainer.evaluate(test_loader, 'test')

def run_all_experiments():
    """Run experiments for all combinations of datasets and models"""
    base_config = {
        'seed': 42,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'conformal_alpha': 0.1,
        'target_layer': 'layer4',
        'dataset_paths': {
            'cytology': 'Datasets/Cytology_Cervical_Cancer',
            'histopathology': 'Datasets/Histopathology_Ovarian_Cancer',
            'mammography': 'Datasets/Mammography_Breast_Cancer'
        },
        'class_mapping': {
            'cytology': {'normal': 0, 'abnormal': 1},
            'histopathology': {'benign': 0, 'malignant': 1},
            'mammography': {'normal': 0, 'tumor': 1}
        }
    }
    
    # Experiment parameters
    datasets = ['cytology', 'histopathology', 'mammography']
    model_configs = [
        {
            'name': 'single_network',
            'model_name': 'resnet50',
            'dropout_rate': 0.0,
            'conformal_method': 'aps'
        },
        {
            'name': 'mc_dropout',
            'model_name': 'resnet50',
            'dropout_rate': 0.5,
            'conformal_method': 'raps'
        },
        {
            'name': 'deep_ensemble',
            'model_name': 'resnet50',
            'dropout_rate': 0.0,
            'conformal_method': 'saps',
            'num_models': 5
        }
    ]
    
    # Results storage
    results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('Outputs', 'results', f'experiment_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Run experiments
    for dataset, model_config in product(datasets, model_configs):
        print(f'\nRunning experiment: Dataset={dataset}, Model={model_config["name"]}')
        
        # Update configuration
        config = base_config.copy()
        config.update({
            'dataset': dataset,
            'model_name': model_config['model_name'],
            'dropout_rate': model_config['dropout_rate'],
            'conformal_method': model_config['conformal_method'],
            'project_name': 'medical_uncertainty',
            'run_name': f'{dataset}_{model_config["name"]}_{timestamp}',
            'output_dir': os.path.join(output_dir, dataset, model_config['name'])
        })
        
        if model_config['name'] == 'deep_ensemble':
            config['num_models'] = model_config['num_models']
        
        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)
        
        try:
            # Run experiment
            metrics = run_experiment(config)
            
            # Store results
            result = {
                'dataset': dataset,
                'model': model_config['name'],
                'timestamp': timestamp,
                **metrics
            }
            results.append(result)
            
            # Save individual experiment results
            with open(os.path.join(config['output_dir'], 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
                
        except Exception as e:
            print(f'Error in experiment: {e}')
            continue
    
    # Save all results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'all_results.csv'), index=False)
    
    # Generate summary statistics
    summary = results_df.groupby(['dataset', 'model']).agg({
        'accuracy': ['mean', 'std'],
        'auc': ['mean', 'std'],
        'ece': ['mean', 'std'],
        'brier_score': ['mean', 'std'],
        'coverage': ['mean', 'std'],
        'average_set_size': ['mean', 'std']
    }).round(4)
    
    summary.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))
    
    # Log final results to W&B
    wandb.init(
        project='medical_uncertainty',
        name=f'final_summary_{timestamp}',
        config=base_config
    )
    
    for dataset in datasets:
        for metric in ['accuracy', 'auc', 'ece', 'brier_score', 'coverage']:
            metric_data = results_df[results_df['dataset'] == dataset]
            wandb.log({
                f'{dataset}/{metric}': wandb.Table(
                    data=[[model, value] for model, value in zip(
                        metric_data['model'],
                        metric_data[metric]
                    )],
                    columns=['model', metric]
                )
            })
    
    wandb.finish()
    
    print('\nExperiments completed! Results saved to:', output_dir)
    return results_df, summary

if __name__ == '__main__':
    results_df, summary = run_all_experiments()
    print('\nSummary of results:')
    print(summary) 