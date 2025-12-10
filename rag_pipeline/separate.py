import os
import json


def filter_empty_descriptions(input_folder, output_folder):
    """
    Read all .jsonl files from input_folder, remove entries with empty descriptions,
    and save filtered files to output_folder.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    total_entries_all = 0
    kept_entries_all = 0
    removed_entries_all = 0

    # Iterate through all .jsonl files in the input folder
    for filename in os.listdir(input_folder):
        # if filename not "meta_Cell_Phones_and_Accessories.jsonl" then skip
        if not filename == "meta_Cell_Phones_and_Accessories.jsonl":
            continue
        if not filename.endswith(".jsonl"):
            continue

        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename)

        total_entries = 0
        kept_entries = 0
        removed_entries = 0

        with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:

            for line in infile:
                if not line.strip():
                    continue

                total_entries += 1

                try:
                    entry = json.loads(line)

                    # Check if description is empty (empty list)
                    has_empty_description = (
                        "description" in entry and 
                        isinstance(entry["description"], list) and 
                        len(entry["description"]) == 0
                    )

                    if not has_empty_description:
                        # Keep this entry - write to output file
                        outfile.write(line)
                        kept_entries += 1
                    else:
                        removed_entries += 1

                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON line in {filename}: {e}")

        print(f"{filename}: Kept {kept_entries}/{total_entries} entries (removed {removed_entries})")

        total_entries_all += total_entries
        kept_entries_all += kept_entries
        removed_entries_all += removed_entries

    print("\n--- TOTALS ---")
    print(f"Total entries across all files: {total_entries_all}")
    print(f"Total entries kept: {kept_entries_all}")
    print(f"Total entries removed: {removed_entries_all}")
    print(f"Percentage removed: {removed_entries_all / total_entries_all:.2%}")


# Example usage:
input_folder = "/scratch/cjpenni/departmental-honors/rag_pipeline/amazon_products"
output_folder = "/scratch/cjpenni/departmental-honors/rag_pipeline/amazon_products/with_desc"

filter_empty_descriptions(input_folder, output_folder)