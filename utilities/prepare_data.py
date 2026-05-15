import pandas as pd
import os
from tqdm import tqdm

class evalPairs():
    def __init__ (self, csv_path):
        self.eval_df = pd.read_csv(csv_path)
    
    def create_eval_groups(self, col_names: type(list)):
        """For supplied columns create evaluation pairs and set the labels for each columns 
        to be used in any graphs"""

        self.group_labels = col_names
        self.eval_groups = []
        for row in tqdm(self.eval_df.to_dict("records")):
            group = []
            for col_name in col_names:
                group.append(row[col_name])
            self.eval_groups.append(group)
    
    def concat_text_insert_ms(self, col_name):
        """Insert ms into a text and keep a record of the offsets. Ms are not added in the middle of eval sentences
        This means each ms will not be exactly 300 tokens, but passim gets a fairer change - is not punished by boundaries"""
        print(f"Building passim input text for: {col_name}")
        ms_offsets = []
        full_text = []
        current_ms = 1
        token_counter = 0
        ms_char_counter = 0
        raw_char_counter = 0
        for row in tqdm(self.eval_df.to_dict("records")):
            text = row[col_name]
            if pd.isna(text):
                ms_offsets.append([0, 0, 0, 0, 0])
                continue

            token_counter += len(text.split())
            
            # Calculate starts and ends
            ms_start = ms_char_counter
            raw_start = raw_char_counter
            char_len = len(text) + 1
            ms_char_counter += char_len
            raw_char_counter += char_len
            ms_end = ms_char_counter
            raw_end = raw_char_counter

            full_text.append(text)
            
            ms_offsets.append([current_ms, ms_start, ms_end, raw_start, raw_end])

            if token_counter > 300:
                full_text.append(current_ms)
                current_ms += 1
                ms_char_counter = 0
                token_counter = 0
        
        # Get the correct zfill
        full_text.append(current_ms)
        digits = len(str(current_ms))
        corrected_ms_text = []
        for text in full_text:
            if type(text) == int:
                text = str(text).zfill(digits)
                text = "ms" + text
            corrected_ms_text.append(text)
        
        full_text = " ".join(corrected_ms_text)
        
        new_cols = []
        data_labels = ["seq", "ms_start", "ms_end", "raw_start", "raw_end"]
        for label in data_labels:
            new_cols.append(f"{col_name}_{label}")
        
        offsets_data = pd.DataFrame(ms_offsets, columns=new_cols)
        self.eval_df = pd.concat([self.eval_df, offsets_data], axis=1)
            
            
        return full_text

    def create_concat_texts(self, col_names: type(list), out_dir):
        for col_name in col_names:
            text = self.concat_text_insert_ms(col_name)
            text_path = os.path.join(out_dir, f"{col_name}.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)
        csv_path = os.path.join(out_dir, "text_with_offsets.csv")
        self.eval_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        


