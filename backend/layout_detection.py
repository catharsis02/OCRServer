import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

OUTPUT_SAVE_PATH = Path(Path.cwd()/"rec_outputs"/"inflated_outputs"/"json")
FINAL_TEXT_PATH = Path.cwd() / "final_text"

class LayoutDetection:
    def __init__(self,outputs_path : str | Path,out_text : str | Path = FINAL_TEXT_PATH,inflated_outputs_path : str | Path = OUTPUT_SAVE_PATH):
        if isinstance(outputs_path,str):
            outputs_path = Path(outputs_path)
        outputs_path.mkdir(parents=True,exist_ok=True)
        self.outputs_path = outputs_path
        self.out_text = Path(out_text)
        self.out_text.mkdir(parents=True,exist_ok=True)
        self.inflated_outputs_path = Path(inflated_outputs_path)
        self.inflated_outputs_path.mkdir(parents=True,exist_ok=True)

    def _overlap(self,box1, box2):
        ax1, ay1, ax2, ay2 = box1
        bx1, by1, bx2, by2 = box2

        return ax1 <= bx2 and ax2 >= bx1 and ay1 <= by2 and ay2 >= by1

    def _load_data_from_json(self,output):
        with open(output,"r",encoding="utf-8") as f:
            return json.load(f)
    

    def _inflate(self,data):
        boxes_and_text = []
        inflated_boxes_and_text = []
        f_median = []
        for item in data:
            boxes_and_text.append({
                "box" : item["box"],
                "text" : item["text"]}
                )
            box = item["box"]
            ys = [int(p[1]) for p in box]
            height = max(ys) - min(ys)
            f_median.append(height)

        median = int(np.median(f_median)) if f_median else 0

        for item in data:
            box = item["box"]
            xs = [int(value[0]) for value in box]
            ys = [int(value[1]) for value in box]
            T = median / 2
            xmin = int((min(xs) - T))
            xmax = int((max(xs) + T))
            ymin = int((min(ys) - T))
            ymax = int((max(ys) + T))
            inflated_boxes_and_text.append({
                "box": [[xmin,ymin,xmax,ymax]],
                "text" : item["text"]
    })
        return inflated_boxes_and_text

    def _transform_data(self,outputs):
        for output in outputs:
            data = self._load_data_from_json(output)
            inflated_boxes_and_text = self._inflate(data)
            with open(self.inflated_outputs_path /f"{output.stem}.json","w",encoding="utf-8") as f:
                json.dump(inflated_boxes_and_text,f,indent=2)
    
    def _find(self,x,parent):
        if parent[x] != x:
            parent[x],parent = self._find(parent[x],parent)
        return parent[x],parent

    def _union(self,a, b,parent,rank):
        pa,parent = self._find(a,parent)
        pb,parent = self._find(b,parent)

        if pa == pb:
            return parent,rank

        if rank[pa] < rank[pb]:
            parent[pa] = pb
        elif rank[pa] > rank[pb]:
            parent[pb] = pa
        else:
            parent[pb] = pa
            rank[pa] += 1
        
        return parent,rank
    
    def _comparator(self,x,boxes):
        return (boxes[x][1]+boxes[x][3])/2

    def predict(self):
        if self.outputs_path.is_file():
            self.outputs = [self.outputs_path]
        else:
            self.outputs = [output for output in self.outputs_path.iterdir() if output.is_file()]
        final_texts = []
        outputs = self.outputs
        self._transform_data(outputs)
        files = [file for file in self.inflated_outputs_path.iterdir()]
        cell_size = 100
        print("Formatting OCR predictions")
        for file in tqdm(files):
            final_texts.append("="*50)
            final_texts.append(f"Page : {file.stem}")
            final_texts.append("="*50)
            grid = {}
            with open(file,"r",encoding="utf-8") as f:
                rec_dict = json.load(f)
            
            boxes = [item["box"][0] for item in rec_dict]
            texts = [item["text"] for item in rec_dict]

            n = len(boxes)

            parent = list(range(n))
            rank = [0] * n
            
            for box_id,box in enumerate(boxes):

                xmin,ymin,xmax,ymax = box
                gx0 = int(xmin) // cell_size
                gx1 = int(xmax) // cell_size
                gy0 = int(ymin) // cell_size
                gy1 = int(ymax) // cell_size

                for i in range(gx0,gx1+1):
                    for j in range(gy0,gy1+1):
                        if (i,j) not in grid:
                            grid[(i,j)] = []
                        grid[(i,j)].append(box_id)
            
            for key in grid:
                x,y = key
                for box_id in grid[key]:
                    for i in range(x - 1, x + 2):
                        for j in range(y - 1, y + 2):

                            if (i, j) not in grid:
                                continue

                            for other_box_id in grid[(i, j)]:
                                if box_id == other_box_id:
                                    continue

                                if self._overlap(boxes[box_id], boxes[other_box_id]):
                                    parent,rank = self._union(box_id, other_box_id,parent,rank)
                
            clusters = dict()

            for box_id in range(len(boxes)):
                x,parent = self._find(box_id,parent)
                if x not in clusters:
                    clusters[x] = []
                clusters[x].append(box_id)

            
            sorted_clusters = sorted(
                clusters.items(),
                key=lambda item: min(boxes[idx][1] for idx in item[1])
            )


            with open(self.out_text  / f"{file.stem}.txt", "w", encoding="utf-8") as f:
                for k, cluster in sorted_clusters:
                    box_lst = sorted(
                        cluster,
                        key=lambda x: self._comparator(x, boxes)
                    )

                    for item in box_lst:
                        final_texts.append(texts[item])
                        f.write(texts[item] + "\n")

                    f.write("\n")
        return final_texts

if __name__ == "__main__":
    lay_dec = LayoutDetection(outputs_path="../rec_output_save_path/json",out_text="../final_text")
    lay_dec.predict()