import pandas as pd
import os

# folder data
folder_path = "data"

# list semua file csv
files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

df_list = []

for file in files:
    file_path = os.path.join(folder_path, file)
    
    try:
        temp_df = pd.read_csv(file_path)
        temp_df['source_file'] = file  # optional (buat tracking)
        df_list.append(temp_df)
        print(f"Loaded: {file}")
    except Exception as e:
        print(f"Error di {file}: {e}")

# gabungkan semua
df = pd.concat(df_list, ignore_index=True)

print("Total data:", len(df))

# simpan hasil
df.to_csv("gabungan.csv", index=False)

print("Selesai! File disimpan sebagai gabungan.csv")