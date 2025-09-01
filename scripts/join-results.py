import pandas as pd

base_file = "index-download/index-{year}.csv"
model_names = ["qwen3", "gpt-oss", "gemini-flash-lite"]#, "gpt-5-nano"]
model_file = "index-classification-{model_name}/index-classification-{model_name}-{year}.csv"
df_list = []
years = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012]
for year in years:
    df = pd.read_csv(base_file.format(year=year))
    for model_name in model_names:
        df_model = pd.read_csv(model_file.format(model_name=model_name, year=year)).add_suffix(f"_{model_name}")
        df_model = df_model.rename(columns={f"filename_{model_name}": "downloadFilename"})
        df = df.merge(df_model, on="downloadFilename", how="left")
    # df.to_csv(f"index-joined/index-joined-{year}.csv", index=False, encoding="utf-8")
    df_list.append(df)

df_joined =pd.concat(df_list)

df_joined.to_csv(f"index-joined/index-joined-{min(years)}-{max(years)}.csv", index=False, encoding="utf-8")

list_out = []

for _, row in df_joined.iterrows():
    is_reinsurance, is_main_contract, is_obligatory, proportional_or_non_proportional, insurance_type = None, None, None, None, None
    if row["is_reinsurance_qwen3"] is not None:
        if row["is_reinsurance_gpt-oss"] == row["is_reinsurance_qwen3"] or row["is_reinsurance_gemini-flash-lite"] == row["is_reinsurance_qwen3"]:
            is_reinsurance = row["is_reinsurance_qwen3"]
        elif row["is_reinsurance_gpt-oss"] is None and row["is_reinsurance_gemini-flash-lite"] is None:
            is_reinsurance = row["is_reinsurance_qwen3"]
    else:
        if row["is_reinsurance_gpt-oss"] is not None:
            if row["is_reinsurance_gemini-flash-lite"] is None or row["is_reinsurance_gemini-flash-lite"] == row["is_reinsurance_gpt-oss"]:
                is_reinsurance = row["is_reinsurance_gpt-oss"]
        elif row["is_reinsurance_gemini-flash-lite"] is not None:
            is_reinsurance = row["is_reinsurance_gemini-flash-lite"]

    if is_reinsurance:
        if row["is_main_contract_qwen3"] is not None:
            if row["is_main_contract_gpt-oss"] == row["is_main_contract_qwen3"] or row["is_main_contract_gemini-flash-lite"] == row["is_main_contract_qwen3"]:
                is_main_contract = row["is_main_contract_qwen3"]
            elif row["is_main_contract_gpt-oss"] is None and row["is_main_contract_gemini-flash-lite"] is None:
                is_main_contract = row["is_main_contract_qwen3"]
        else:
            if row["is_main_contract_gpt-oss"] is not None:
                if row["is_main_contract_gemini-flash-lite"] is None or row["is_main_contract_gemini-flash-lite"] == row["is_main_contract_gpt-oss"]:
                    is_main_contract = row["is_main_contract_gpt-oss"]
            elif row["is_main_contract_gemini-flash-lite"] is not None:
                is_main_contract = row["is_main_contract_gemini-flash-lite"]

    if is_main_contract:
        if row["is_obligatory_qwen3"] is not None:
            if row["is_obligatory_gpt-oss"] == row["is_obligatory_qwen3"] or row["is_obligatory_gemini-flash-lite"] == row["is_obligatory_qwen3"]:
                is_obligatory = row["is_obligatory_qwen3"]
            elif row["is_obligatory_gpt-oss"] is None and row["is_obligatory_gemini-flash-lite"] is None:
                is_obligatory = row["is_obligatory_qwen3"]
        else:
            if row["is_obligatory_gpt-oss"] is not None:
                if row["is_obligatory_gemini-flash-lite"] is None or row["is_obligatory_gemini-flash-lite"] == row["is_obligatory_gpt-oss"]:
                    is_obligatory = row["is_obligatory_gpt-oss"]
            elif row["is_obligatory_gemini-flash-lite"] is not None:
                is_obligatory = row["is_obligatory_gemini-flash-lite"]

    if is_main_contract:
        if row["proportional_or_non_proportional_qwen3"] is not None:
            if row["proportional_or_non_proportional_gpt-oss"] == row["proportional_or_non_proportional_qwen3"] or row["proportional_or_non_proportional_gemini-flash-lite"] == row["proportional_or_non_proportional_qwen3"]:
                proportional_or_non_proportional = row["proportional_or_non_proportional_qwen3"]
            elif row["proportional_or_non_proportional_gpt-oss"] is None and row["proportional_or_non_proportional_gemini-flash-lite"] is None:
                proportional_or_non_proportional = row["proportional_or_non_proportional_qwen3"]
        else:
            if row["proportional_or_non_proportional_gpt-oss"] is not None:
                if row["proportional_or_non_proportional_gemini-flash-lite"] is None or row["proportional_or_non_proportional_gemini-flash-lite"] == row["proportional_or_non_proportional_gpt-oss"]:
                    proportional_or_non_proportional = row["proportional_or_non_proportional_gpt-oss"]
            elif row["proportional_or_non_proportional_gemini-flash-lite"] is not None:
                proportional_or_non_proportional = row["proportional_or_non_proportional_gemini-flash-lite"]

    if is_main_contract:
        if row["insurance_type_qwen3"] is not None:
            if row["insurance_type_gpt-oss"] == row["insurance_type_qwen3"] or row["insurance_type_gemini-flash-lite"] == row["insurance_type_qwen3"]:
                insurance_type = row["insurance_type_qwen3"]
            elif row["insurance_type_gpt-oss"] is None and row["insurance_type_gemini-flash-lite"] is None:
                insurance_type = row["insurance_type_qwen3"]
        else:
            if row["insurance_type_gpt-oss"] is not None:
                if row["insurance_type_gemini-flash-lite"] is None or row["insurance_type_gemini-flash-lite"] == row["insurance_type_gpt-oss"]:
                    insurance_type = row["insurance_type_gpt-oss"]
            elif row["insurance_type_gemini-flash-lite"] is not None:
                insurance_type = row["insurance_type_gemini-flash-lite"]

    row_dict = row.to_dict() | {
        "is_reinsurance": is_reinsurance,
        "is_main_contract": is_main_contract,
        "is_obligatory": is_obligatory,
        "proportional_or_non_proportional": proportional_or_non_proportional,
        "insurance_type": insurance_type
    }

    list_out.append(row_dict)

df_out = pd.DataFrame(list_out)
df_out.to_csv(f"index-joined/index-joined-final-{min(years)}-{max(years)}.csv", index=False, encoding="utf-8")