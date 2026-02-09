#!/usr/bin/env python3
# Convert the TREC Health Misinformation Topics into TSV files

import xml.etree.ElementTree as ET
import os

def main(topics, fields):
    xml = ET.parse(topics)
    root = xml.getroot()
    out_folder = "./queries"
    os.makedirs(out_folder, exist_ok=True)
    basename = os.path.basename(topics)
    for field in fields:
        out_field_name = basename.split(".")[0]+f"-{field}.tsv"
        out_field_file = os.path.join(out_folder, out_field_name)
        with open(out_field_file, 'w') as fout:
            for child in root:
                number = child.findall('number')[0].text
                field_column = child.findall(field)[0].text
                fout.write(f"{number}\t{field_column}\n")

if __name__ == '__main__':
    topics2020 = './trec-misinfo-resources/2020/topics/misinfo-2020-topics.xml'
    main(topics2020, ['title', 'description'])

    topics2021 = './trec-misinfo-resources/2021/topics/misinfo-2021-topics.xml'
    main(topics2021, ['query', 'description'])

    topics2022 = './trec-misinfo-resources/2022/topics/misinfo-2022-topics.xml'
    main(topics2022, ['query', 'question'])
