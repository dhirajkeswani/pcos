import pandas as pd

df = pd.read_excel(r"C:\Users\dhiru\Downloads\Dhiraj_1\KaggleDataset\PCOS_data_without_infertility.xlsx", sheet_name="Full_new")

df.to_csv(r"C:\Users\dhiru\Downloads\Dhiraj_1\KaggleDataset\PCOS_data_without_infertility.csv", index=False)

print("done")