import yaml
from pathlib import Path
yml_file = Path.cwd() / "models" / "PP-OCRv5_server_rec_infer" / "inference.yml"
with open(yml_file,"r+",encoding="utf-8") as f:
    try:
        obj = yaml.load(f,Loader=yaml.CLoader)
    except Exception:
        obj = yaml.load(f,Loader=yaml.Loader)
obj = dict(obj)
character_lst = obj["PostProcess"]["character_dict"]
print(character_lst)
print(len(character_lst)) # 18383
character_str : str = "".join(character_lst)
ch_lst = [i for i in character_str]
print(ch_lst) # 18393
print(len(ch_lst))
multi = [ch for ch in character_lst if len(ch) != 1]
print(len(multi))

with open("character_str.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(character_lst))
