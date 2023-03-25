import json
import os
path = "/root/Alpaca/iflmwvlt/formated_cot_data/"
file_list = os.listdir(path)
final_list = []
for f_name in file_list:
    with open(path+f_name,"r") as f:
        a = json.load(f)
        final_list = final_list + a

assert len(final_list) == 74771

fw=open("/root/Alpaca/iflmwvlt/CoT_data.json", "w", encoding="utf8")
json.dump(final_list, fw, ensure_ascii=False)
