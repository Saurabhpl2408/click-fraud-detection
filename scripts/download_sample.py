import pandas as pd
import numpy as np

# Create synthetic sample data for testing
np.random.seed(42)

n_rows = 100000
data = {
    'ip': np.random.randint(1, 100000, n_rows),
    'app': np.random.randint(1, 500, n_rows),
    'device': np.random.randint(0, 3000, n_rows),
    'os': np.random.randint(1, 800, n_rows),
    'channel': np.random.randint(1, 500, n_rows),
    'click_time': pd.date_range('2017-11-06', periods=n_rows, freq='1s'),
    'is_attributed': np.random.choice([0, 1], n_rows, p=[0.998, 0.002])
}

df = pd.DataFrame(data)
df.to_csv('data/raw/train_sample.csv', index=False)
print(f"✅ Created sample dataset: {len(df)} rows")
print(f"Fraud rate: {df['is_attributed'].mean()*100:.2f}%")
