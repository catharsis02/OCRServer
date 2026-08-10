from pathlib import Path
import numpy as np
import re
class CTCDecodePostProcess:
    def __init__(self,character_list=None,use_space_char=True):
        if character_list is not None:
            self.character_list = character_list
        else:
            try:
                file = Path(__file__).parent.parent / "character_str.txt"
                with open(file,"r",encoding="utf-8") as f:
                    self.character_list=list(f.read().splitlines())
            except FileNotFoundError:
                self.character_list = list("0123456789abcdefghijklmnopqrstuvwxyz")
        
        if use_space_char:
            self.character_list.append(" ")
        
        self.character_list = ["blank"] + self.character_list
        self.char_dict = {}
        for i,ch in enumerate(self.character_list):
            self.char_dict[i] = ch

         
    def __call__(self,pred,return_word_box=False,**kwargs):
        preds = pred
        if preds.ndim == 2:
            preds = np.expand_dims(preds, axis=0)
        preds_idx = np.argmax(preds,axis=-1)
        preds_prob = np.max(preds,axis=-1)
        

        text = self.decode(preds_idx=preds_idx,preds_prob=preds_prob,remove_duplicates=True,return_word_box=return_word_box)
        if return_word_box:
            for rec_idx, rec in enumerate(text):
                wh_ratio = kwargs["wh_ratio_list"][rec_idx]
                max_wh_ratio = kwargs["max_wh_ratio"]
                rec[2][0] = rec[2][0] * (wh_ratio / max_wh_ratio)
        texts,scores = [],[]
        for t in text:
            texts.append(t[0] if len(t) <= 2 else (t[0], t[2]))
            scores.append(t[1])
        return texts,scores
    
    def get_word_box(self,text,selection):
        state = None
        word_content = []
        word_col_content = []
        word_list = []
        word_col_list = []
        states = []
        valid_col = np.where(selection)[0]

        for c_i, char in enumerate(text):
            if "\u4e00" <= char <= "\u9fff":
                c_state = "cn"
            elif bool(re.search("[a-zA-Z0-9]", char)):
                c_state = "en&num"
            else:
                c_state = "symbol"

            if (
                char == "."
                and state == "en&num"
                and c_i + 1 < len(text)
                and bool(re.search("[0-9]", text[c_i + 1]))
            ):
                c_state = "en&num"

            if char == "-" and state == "en&num":
                c_state = "en&num"

            if state is None:
                state = c_state

            if state != c_state:
                if len(word_content) != 0:
                    word_list.append(word_content)
                    word_col_list.append(word_col_content)
                    states.append(state)
                    word_content = []
                    word_col_content = []
                state = c_state


            word_content.append(char)
            word_col_content.append(int(valid_col[c_i]))

        if len(word_content) != 0:
            word_list.append(word_content)
            word_col_list.append(word_col_content)
            states.append(state)

        return word_list, word_col_list, states


    def decode(self,preds_idx,preds_prob,remove_duplicates=False,return_word_box=False):
        result_list=[]
        ignored_tokens = [0]
        batch_size = len(preds_idx)
        for batch_idx in range(batch_size):
            
            seq = preds_idx[batch_idx].squeeze()
            selection = np.ones(seq.shape, dtype=bool)

            if remove_duplicates:
                selection[1:] = preds_idx[batch_idx][1:] != preds_idx[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection = selection & (preds_idx[batch_idx] != ignored_token)
        
            char_list = [self.character_list[text_id] for text_id in preds_idx[batch_idx][selection]]

            if preds_prob is not None:
                confidence_list = preds_prob[batch_idx][selection]
            else:
                confidence_list = [1] * len(char_list)
            if len(confidence_list) == 0:
                confidence_list=[0]
            
            text = "".join(char_list)
            if return_word_box:
                word_list, word_col_list, state_list = self.get_word_box(
                    text, selection
                )
                result_list.append(
                    (
                        text,
                        np.mean(confidence_list).tolist(),
                        [
                            len(preds_idx[batch_idx]),
                            word_list,
                            word_col_list,
                            state_list,
                        ],
                    )
                )
            else:

                result_list.append((text,np.mean(confidence_list).tolist()))
        return result_list
