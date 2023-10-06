import os
import csv

# insert location of annotation .txt files
text_files_dir = '/Users/wilsco/Downloads/mirflickr25k_annotations_v080'

categories = {}
unique_cats = set()
skip_file = ['README.txt']

for filename in os.listdir(text_files_dir):
    if filename.endswith('.txt') and filename not in skip_file:
        file_path = os.path.join(text_files_dir, filename)
        category_name = os.path.splitext(filename)[0]
        if category_name.endswith('_r1'):
            category_name = category_name.split("_")[0]

        # open, read file
        print(f"Opening {file_path}")
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            line = line.strip()
            if category_name not in categories:
                categories[f"im{line}.jpg"] = set()
                print(f"Adding {line} to dictionary of categories")
            categories[f"im{line}.jpg"].add(category_name)
            unique_cats.add(category_name)

unique_files = list(categories.keys())

csv_filename = 'labels.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # add col names
    header_row = ['File'] + sorted(unique_cats)
    writer.writerow(header_row)
    
    images_in_multiple_categories = []

    for file_name in unique_files:
        file_row = [file_name]

        for category in sorted(unique_cats):
            if category in categories.get(file_name, set()):
                file_row.append(1)
            else:
                file_row.append(0)
        
        writer.writerow(file_row)

        if list(file_row[1:]).count(1) > 1:
            images_in_multiple_categories.append(file_name)

print("Finished label.csv")
for image_name in images_in_multiple_categories:
    print(image_name)