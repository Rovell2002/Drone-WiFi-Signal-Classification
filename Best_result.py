import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

def load_history_files(directory):
    history_list = []
    file_names = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'rb') as file:
                history_list.append(pickle.load(file))
                file_names.append(file_name)
    return history_list, file_names

def calculate_composite_score(history):
    max_val_accuracy = max(history.get('val_accuracy', [0]))
    max_f1_score = max(history.get('f1_score', [0]))
    max_precision = max(history.get('precision', [0]))
    max_recall = max(history.get('recall', [0]))
    min_val_loss = min(history.get('val_loss', [float('inf')]))

    composite_score = (
        max_val_accuracy + max_f1_score + max_precision + max_recall - min_val_loss
    )
    return composite_score

def analyze_histories(histories, file_names):
    results = []
    for history, file_name in zip(histories, file_names):
        max_train_accuracy = max(history.get('train_accuracy', [0]))
        max_val_accuracy = max(history.get('val_accuracy', [0]))
        max_f1_score = max(history.get('f1_score', [0]))
        max_precision = max(history.get('precision', [0]))
        max_recall = max(history.get('recall', [0]))
        min_train_loss = min(history.get('train_loss', [float('inf')]))
        min_val_loss = min(history.get('val_loss', [float('inf')]))
        model_config = history.get('config', {})
        composite_score = calculate_composite_score(history)
        results.append((history, model_config, max_train_accuracy, max_val_accuracy, max_f1_score, max_precision, max_recall, min_train_loss, min_val_loss, file_name, composite_score))
    
    # Sort results by composite score in descending order
    results.sort(key=lambda x: x[-1], reverse=True)
    # Return the best result
    return results[0] if results else None

def plot_comparison(best_model):
    if not best_model:
        print("No model found.")
        return
    
    metrics = ['train_accuracy', 'val_accuracy', 'train_loss', 'val_loss', 'f1_score', 'precision', 'recall']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        history = best_model[0]
        if metric in history:
            plt.plot(history[metric], label=f'Best Model')
        plt.title(f'Comparison of {metric.replace("_", " ").title()}')
        plt.xlabel('Epochs')
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.grid(True)
        plt.savefig(f'comparison_{metric}.png')
        plt.show()

def main():
    # Specify the directory where your history files are stored
    directory = r'C:\Users\lolaf\Desktop\Thesis_code\Results\LSTM\History'
    
    histories, file_names = load_history_files(directory)
    best_model = analyze_histories(histories, file_names)
    
    if best_model:
        history, best_model_config, best_train_accuracy, best_val_accuracy, best_f1_score, best_precision, best_recall, best_train_loss, best_val_loss, best_file_name, composite_score = best_model
        
        print("\nBest Model:")
        print("Best Model Configuration:")
        print(best_model_config)
        print("Best Training Accuracy:")
        print(best_train_accuracy)
        print("Best Validation Accuracy:")
        print(best_val_accuracy)
        print("Best F1 Score:")
        print(best_f1_score)
        print("Best Precision:")
        print(best_precision)
        print("Best Recall:")
        print(best_recall)
        print("Best Training Loss:")
        print(best_train_loss)
        print("Best Validation Loss:")
        print(best_val_loss)
        print("Composite Score:")
        print(composite_score)
        print("File Name:")
        print(best_file_name)
    
    plot_comparison(best_model)

if __name__ == "__main__":
    main()