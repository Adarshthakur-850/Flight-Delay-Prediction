import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def perform_eda(df):
    print("Performing EDA...")
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    # Delay Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='arrival_delay', bins=50, kde=True)
    plt.title("Arrival Delay Distribution")
    plt.xlim(-60, 180) # Limit x-axis to focus on common delays
    plt.savefig("plots/delay_distribution.png")
    plt.close()
    
    # Airline-wise Delay
    if 'airline' in df.columns and 'is_delayed' in df.columns:
        plt.figure(figsize=(12, 6))
        # Compute delay rate per airline
        delay_rates = df.groupby('airline')['is_delayed'].mean().sort_values(ascending=False)
        sns.barplot(x=delay_rates.index, y=delay_rates.values)
        plt.title("Delay Rate by Airline")
        plt.ylabel("Proportion of Delayed Flights")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("plots/airline_delay_rate.png")
        plt.close()
        
    # Hourly Delay Pattern
    if 'dep_hour' in df.columns and 'is_delayed' in df.columns:
        plt.figure(figsize=(10, 6))
        hourly_delay = df.groupby('dep_hour')['is_delayed'].mean()
        sns.lineplot(x=hourly_delay.index, y=hourly_delay.values, marker='o')
        plt.title("Delay Probability by Hour of Day")
        plt.ylabel("Probability of Delay")
        plt.xlabel("Hour of Day")
        plt.grid(True)
        plt.savefig("plots/hourly_delay_pattern.png")
        plt.close()
