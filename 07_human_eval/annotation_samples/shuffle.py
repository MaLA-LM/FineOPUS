import json
import random

langs = ["ara_Arab", "ben_Beng", "zho_Hans", "fin_Latn"]
# use placeholder in the file names to format them with the language code
files = [f"sample_FineOPUS-Filtered-Stage4_{{L}}_eng_Latn_100.jsonl", f"sample_FineOPUS-Original_unique_{{L}}_eng_Latn_100.jsonl"]

for lang in langs:
    data = []
    for file in files:
        with open(file.format(L=lang), "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # always make lang2_code eng_Latn
                if item.get("lang2_code") != "eng_Latn":
                    # swap lang1_code and lang2_code
                    item["lang2_code"], item["lang1_code"] = item["lang1_code"], item["lang2_code"]
                    # swap lang1_text and lang2_text
                    item["lang2_text"], item["lang1_text"] = item["lang1_text"], item["lang2_text"]
                data.append(item)

    random.shuffle(data)
    with open("shuffled_data_{L}.jsonl".format(L=lang), "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
