import pandas as pd
import os
from typing import Dict

class BOMReader:
    def __init__(self, bom_file: str = "BOM.xlsx"):
        self.bom_file = bom_file
        self.bom_data = None
        self.valid_class_names = set()
        self._load_bom()

    def _load_bom(self) -> None:
        if not os.path.exists(self.bom_file):
            raise FileNotFoundError(f"BOM file not found: {self.bom_file}")
        
        self.bom_data = pd.read_excel(self.bom_file)
        self.valid_class_names = set(self.bom_data['Class_Name'].unique())

    def get_part_info(self, class_name: str) -> Dict[str, str]:
        if class_name not in self.valid_class_names:
            return {
                'program': 'Unknown',
                'part_number': 'Unknown',
                'description': 'Unknown'
            }

        try:
            part_info = self.bom_data[self.bom_data['Class_Name'] == class_name].iloc[0]
            return {
                'program': part_info['Program'],
                'part_number': part_info['Part_Number'],
                'description': part_info['Description']
            }
        except (IndexError, KeyError):
            return {
                'program': 'Unknown',
                'part_number': 'Unknown',
                'description': 'Unknown'
            }