from pathlib import Path
import pandas as pd
import re
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import List


def get_image_id_and_attrs_from_str(str_: str, file_suffix: str = ".jpg") -> List[str]:
    key_idx_end = re.search(file_suffix, str_).span()[-1]
    image_id = str_[:key_idx_end-len(".jpg")]
    attrs = str_[key_idx_end+1:]
    attrs_list = [image_id]
    attrs_list.extend([att for att in attrs.split(" ") if att not in [""]])

    return attrs_list


def convert_celeba_attr_txt_to_csv(attr_txt_file: Path, output_csv_path: Path):
    with open(str(attr_txt_file.absolute()), 'r') as f:
        content = [l.split("\n")[0] for l in f.readlines()]

    attrs_list = []
    print("***EXTRACTING ATTRIBUTES***")
    attrs_list.extend(Parallel(n_jobs=4)(delayed(get_image_id_and_attrs_from_str)(file_line) for file_line in content))
    print("***FINISHED EXTRACTING ATTRIBUTES***")
    del content
    print("***UPDATING DATAFRAME***")
    df = pd.DataFrame(
        columns=[
            "image_id",
            "5_o_Clock_Shadow",
            "Arched_Eyebrows",
            "Attractive",
            "Bags_Under_Eyes",
            "Bald",
            "Bangs",
            "Big_Lips",
            "Big_Nose",
            "Black_Hair",
            "Blond_Hair",
            "Blurry",
            "Brown_Hair",
            "Bushy_Eyebrows",
            "Chubby",
            "Double_Chin",
            "Eyeglasses",
            "Goatee",
            "Gray_Hair",
            "Heavy_Makeup",
            "High_Cheekbones",
            "Male",
            "Mouth_Slightly_Open",
            "Mustache",
            "Narrow_Eyes",
            "No_Beard",
            "Oval_Face",
            "Pale_Skin",
            "Pointy_Nose",
            "Receding_Hairline",
            "Rosy_Cheeks",
            "Sideburns",
            "Smiling",
            "Straight_Hair",
            "Wavy_Hair",
            "Wearing_Earrings",
            "Wearing_Hat",
            "Wearing_Lipstick",
            "Wearing_Necklace",
            "Wearing_Necktie",
            "Young",
        ]
    )
    for attrs in tqdm(attrs_list):
        df.loc[len(df)] = attrs
    print("***FINISHED UPDATING DATAFRAME***")

    df.to_csv(str(output_csv_path.absolute()))


def main():
    attr_txt_file = Path("C:/Users/Chris/Documents/NCSU-Graduate/Courses/ECE792/Project/datasets/realimages/celeba/list_attr_celeba_prepped_for_csv.txt")
    output_csv_path = Path("C:/Users/Chris/Documents/NCSU-Graduate/Courses/ECE792/Project/datasets/realimages/celeba/celeba_attr.csv")
    convert_celeba_attr_txt_to_csv(attr_txt_file, output_csv_path)


if __name__ == '__main__':
    main()