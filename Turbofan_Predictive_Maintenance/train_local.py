import pandas as pd
from xgboost import XGBRegressor
import joblib

print("1. Training Data Load ho raha hai...")
columns = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
train_df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=columns)

print("2. Data Preprocessing...")
train_max = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
train_max.rename(columns={'time_in_cycles': 'max_cycle'}, inplace=True)
train_df = train_df.merge(train_max, on=['unit_number'], how='left')
train_df['RUL'] = train_df['max_cycle'] - train_df['time_in_cycles']
train_df['RUL'] = train_df['RUL'].clip(upper=125)

drop_cols = ['setting_1', 'setting_2', 'setting_3', 'sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
train_df.drop(columns=drop_cols, inplace=True)
features = [col for col in train_df.columns if col.startswith('sensor')]

print("3. Model Train ho raha hai (Local Machine par)...")
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(train_df[features], train_df['RUL'])

print("4. Model Save ho raha hai...")
joblib.dump(model, 'turbofan_xgb_model.pkl')
print("✅ SUCCESS! Naya 'turbofan_xgb_model.pkl' aapke laptop par ban gaya hai!")