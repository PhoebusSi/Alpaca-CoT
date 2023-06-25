import json
import openai
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

openai.api_key = "XXXXXXXXXX"

d = pd.read_json("./raw/belle/eval_prompt.json")
m = {t["class"]:t["prompt"] for t in d.to_dict(orient="records")}

fl = os.listdir(f"./result/belle")
print(fl)

for fa in fl:
    print(fa)
    d = pd.read_json(f"./result/belle/{fa}")
    d["X"] = d["P"].apply(lambda x:x.split("### Response:")[1].split("###")[0].strip())
    d["R"] = ""
    d["S"] = ""
    d["P"] = ""

    wait = d[d["S"]==""].index[:]
    retry = 0
    while len(wait)!= 0:
        for i in tqdm(wait):
            item = d.iloc[i]
            inputPrompt = ""
            mc = m.get(item["C"],"XXXXX")
            if mc == "XXXXX":
                print(i, mc, item)

            if item["C"] in ["generation", "brainstorming", "rewrite"]:
                inputPrompt = "%s %s 模型回答： '%s'。请针对模型回答给出得分，顺便给出理由："%(mc, item["Q"], item["X"])
            else:
                inputPrompt = "%s %s 标准回答： %s 模型回答： '%s'。请针对模型回答给出得分，顺便给出理由："%(mc, item["Q"], item["A"], item["X"])
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0301",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": inputPrompt},
                    ],
                )
                d.loc[i,"R"] = inputPrompt
                d.loc[i,"S"] = response["choices"][0]["message"]["content"].strip()
                print(d.iloc[i], flush=True)
            except Exception as e:
                print(i,e)
        wait = d[d["S"]==""].index[:]
        if retry > 5:
            break
        retry = retry + 1
    print(fa, wait)
    with open(f"./result/gpt/{fa}", 'w', encoding = 'utf-8') as f:
        json.dump(d.to_dict(orient="records"), f, ensure_ascii=False, allow_nan=False, indent=4)     

