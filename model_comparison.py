import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from typing import List, Dict
import json
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

class ModelComparison:
    def __init__(self):
        self.models = ['mt5', 'xlmr', 'tinybert']
        self.results = {}
        
    def load_results(self, mt5_file: str, xlmr_file: str, tinybert_file: str):
        """Load results from all models"""
        try:
            self.results['mt5'] = pd.read_csv(mt5_file, sep='\t')
            self.results['xlmr'] = pd.read_csv(xlmr_file, sep='\t')
            self.results['tinybert'] = pd.read_csv(tinybert_file, sep='\t')
            
            # Convert string representations to actual lists/dicts
            for model in self.models:
                self.results[model]['intents'] = self.results[model]['intents'].apply(eval)
                self.results[model]['slots'] = self.results[model]['slots'].apply(eval)
                
        except Exception as e:
            print(f"Error loading results: {str(e)}")
            raise

    def analyze_intent_distribution(self):
        """Analyze and visualize intent distribution across models"""
        intent_counts = {model: Counter() for model in self.models}
        
        # Count intents for each model
        for model in self.models:
            for intents in self.results[model]['intents']:
                for intent in intents:
                    intent_counts[model][intent['intent']] += 1
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        
        # Plot intent distribution
        for i, (model, counts) in enumerate(intent_counts.items()):
            plt.subplot(1, 3, i+1)
            intents = list(counts.keys())
            frequencies = list(counts.values())
            
            plt.bar(intents, frequencies)
            plt.title(f'{model.upper()} Intent Distribution')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('intent_distribution.png')
        plt.close()
        
        return intent_counts

    def analyze_confidence_scores(self):
        """Analyze and visualize confidence score distributions"""
        confidence_scores = {model: [] for model in self.models}
        
        # Collect confidence scores
        for model in self.models:
            for intents in self.results[model]['intents']:
                for intent in intents:
                    confidence_scores[model].append(intent['confidence'])
        
        # Create violin plot
        plt.figure(figsize=(10, 6))
        data = [confidence_scores[model] for model in self.models]
        
        plt.violinplot(data, showmeans=True)
        plt.xticks(range(1, len(self.models) + 1), [m.upper() for m in self.models])
        plt.ylabel('Confidence Score')
        plt.title('Confidence Score Distribution by Model')
        
        plt.savefig('confidence_distribution.png')
        plt.close()
        
        return confidence_scores

    def analyze_slot_types(self):
        """Analyze and visualize slot type distribution"""
        slot_counts = {model: Counter() for model in self.models}
        
        # Count slot types
        for model in self.models:
            for slots in self.results[model]['slots']:
                for slot in slots:
                    slot_counts[model][slot['type']] += 1
        
        # Create heatmap
        all_types = set()
        for counts in slot_counts.values():
            all_types.update(counts.keys())
        
        matrix = []
        for model in self.models:
            row = [slot_counts[model][slot_type] for slot_type in all_types]
            matrix.append(row)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(matrix, 
                   xticklabels=list(all_types),
                   yticklabels=[m.upper() for m in self.models],
                   annot=True, 
                   fmt='d',
                   cmap='YlOrRd')
        
        plt.title('Slot Type Distribution Across Models')
        plt.xticks(rotation=45, ha='right')
        
        plt.savefig('slot_distribution.png')
        plt.close()
        
        return slot_counts

    def create_interactive_dashboard(self):
        """Create an interactive HTML dashboard using Plotly"""
        # Intent Analysis
        intent_counts = {model: Counter() for model in self.models}
        for model in self.models:
            for intents in self.results[model]['intents']:
                for intent in intents:
                    intent_counts[model][intent['intent']] += 1
        
        # Create intent distribution plot
        fig_intents = go.Figure()
        for model in self.models:
            fig_intents.add_trace(go.Bar(
                name=model.upper(),
                x=list(intent_counts[model].keys()),
                y=list(intent_counts[model].values())
            ))
        
        fig_intents.update_layout(
            title='Intent Distribution by Model',
            barmode='group',
            xaxis_tickangle=-45
        )
        
        # Confidence Score Analysis
        confidence_data = []
        for model in self.models:
            for intents in self.results[model]['intents']:
                for intent in intents:
                    confidence_data.append({
                        'model': model.upper(),
                        'confidence': intent['confidence'],
                        'intent': intent['intent']
                    })
        
        df_confidence = pd.DataFrame(confidence_data)
        fig_confidence = px.violin(
            df_confidence, 
            x='model', 
            y='confidence',
            title='Confidence Score Distribution'
        )
        
        # Slot Analysis
        slot_counts = {model: Counter() for model in self.models}
        for model in self.models:
            for slots in self.results[model]['slots']:
                for slot in slots:
                    slot_counts[model][slot['type']] += 1
        
        # Create slot distribution plot
        fig_slots = go.Figure()
        for model in self.models:
            fig_slots.add_trace(go.Bar(
                name=model.upper(),
                x=list(slot_counts[model].keys()),
                y=list(slot_counts[model].values())
            ))
        
        fig_slots.update_layout(
            title='Slot Type Distribution by Model',
            barmode='group',
            xaxis_tickangle=-45
        )
        
        # Save plots to HTML
        with open('model_comparison.html', 'w') as f:
            f.write('<html><body>')
            f.write('<h1>Model Comparison Dashboard</h1>')
            f.write(fig_intents.to_html(full_html=False))
            f.write(fig_confidence.to_html(full_html=False))
            f.write(fig_slots.to_html(full_html=False))
            f.write('</body></html>')

    def generate_summary_report(self):
        """Generate a detailed summary report"""
        report = {
            'total_samples': len(self.results[self.models[0]]),
            'models': {}
        }
        
        for model in self.models:
            model_data = self.results[model]
            
            # Calculate statistics
            avg_intents = np.mean([len(intents) for intents in model_data['intents']])
            avg_confidence = np.mean([
                intent['confidence']
                for intents in model_data['intents']
                for intent in intents
            ])
            avg_slots = np.mean([len(slots) for slots in model_data['slots']])
            
            report['models'][model] = {
                'average_intents_per_sample': float(avg_intents),
                'average_confidence': float(avg_confidence),
                'average_slots_per_sample': float(avg_slots)
            }
        
        # Save report
        with open('comparison_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

if __name__ == "__main__":
    # Initialize comparison
    comparison = ModelComparison()
    
    # Load results
    comparison.load_results(
        r"C:\Users\badug\OneDrive\Desktop\outputs_major\mt5_results.tsv",
        r"C:\Users\badug\OneDrive\Desktop\outputs_major\xlmr_results.tsv",
        r"C:\Users\badug\OneDrive\Desktop\outputs_major\tinybert_results.tsv"
    )
    
    # Generate analyses
    intent_counts = comparison.analyze_intent_distribution()
    confidence_scores = comparison.analyze_confidence_scores()
    slot_counts = comparison.analyze_slot_types()
    
    # Create interactive dashboard
    comparison.create_interactive_dashboard()
    
    # Generate summary report
    report = comparison.generate_summary_report()
    
    print("\nAnalysis complete! Generated files:")
    print("- intent_distribution.png")
    print("- confidence_distribution.png")
    print("- slot_distribution.png")
    print("- model_comparison.html")
    print("- comparison_report.json") 