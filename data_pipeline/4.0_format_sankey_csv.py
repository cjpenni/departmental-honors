import json
import csv
from pathlib import Path
from typing import List, Dict

def convert_json_to_csv_fixed(rag_json_path: str, cat_json_path: str, output_csv_path: str) -> None:
    """
    Convert data from new_inferences_rag.json and inferences_with_cat.json 
    to CSV format matching user_data_extended_v8.csv structure.
    
    This version properly handles:
    - RAG data: Only has inference, uncommonness, sensitivity, and explanation
    - Cat data: Has all fields including category, activity, reason
    
    For RAG data, derives category/activity/reason from the inference text.
    
    Args:
        rag_json_path: Path to new_inferences_rag.json
        cat_json_path: Path to inferences_with_cat.json
        output_csv_path: Output CSV file path
    """
    
    # Load JSON files
    print(f"Loading {rag_json_path}...")
    with open(rag_json_path, 'r', encoding='utf-8') as f:
        rag_data = json.load(f)
    
    print(f"Loading {cat_json_path}...")
    with open(cat_json_path, 'r', encoding='utf-8') as f:
        cat_data = json.load(f)
    
    def extract_keywords(text: str, max_keywords: int = 5) -> str:
        """Extract keywords from inference text for inference_key field"""
        if not text:
            return "[]"
        
        stop_words = {'the', 'a', 'an', 'in', 'of', 'to', 'and', 'or', 'is', 'are', 'on', 'at', 'for', 'interested'}
        words = text.lower().split()
        keywords = [w.strip('.,!?') for w in words 
                   if len(w) > 2 and w.lower() not in stop_words]
        keywords = keywords[:max_keywords]
        
        return str(keywords).replace("'", '"')
    
    def flatten_rag_entries(json_data: List[Dict]) -> List[Dict]:
        """
        Flatten RAG JSON entries - these are missing category/activity/reason.
        Each inference is paired with all final_product_recommendations.
        """
        rows = []

        for entry in json_data:
            combined_cols = entry.get('combined_cols', [])
            combined_col_str = ' AND '.join(combined_cols) if combined_cols else 'N/A'

            gpt_output = entry.get('gpt_output', {})
            inferences_list = gpt_output.get('inferences', [])
            recommendations_list = gpt_output.get('final_product_recommendations', [])

            overall_product = ''
            if recommendations_list:
                overall_product = recommendations_list[0].get('title', '')

            for inference_obj in inferences_list:
                inference_text = inference_obj.get('inference', '')
                explanation = inference_obj.get('explanation', '')

                # Extract category from inference text
                category_hint = inference_text.split(' in ')[-1] if ' in ' in inference_text else inference_text
                category_hint = category_hint.split(' and ')[-1] if ' and ' in category_hint else category_hint
                category = category_hint.strip() if category_hint else 'General Interest'
                if category.lower().startswith('interested'):
                    category = 'General Interest'
                else:
                    category = ' '.join(word.capitalize() for word in category.split())

                # Pair this inference with **all recommendations**
                if recommendations_list:
                    for rec in recommendations_list:
                        row = {
                            'combined_col': combined_col_str,
                            'inference': inference_text,
                            'inference_key': extract_keywords(inference_text),
                            'category': category,
                            'activity': inference_text,
                            'reason': explanation,
                            'recommended_product': rec.get('title', ''),
                            'uncommonness': str(inference_obj.get('uncommonness', '')),
                            'sensitivity': str(inference_obj.get('sensitivity', '')),
                            'overall_recommended_product': overall_product
                        }
                        rows.append(row)
                else:
                    # No recommendations, still add row
                    row = {
                        'combined_col': combined_col_str,
                        'inference': inference_text,
                        'inference_key': extract_keywords(inference_text),
                        'category': category,
                        'activity': inference_text,
                        'reason': explanation,
                        'recommended_product': '',
                        'uncommonness': str(inference_obj.get('uncommonness', '')),
                        'sensitivity': str(inference_obj.get('sensitivity', '')),
                        'overall_recommended_product': overall_product
                    }
                    rows.append(row)

        return rows
    
    def flatten_cat_entries(json_data: List[Dict]) -> List[Dict]:
        """
        Flatten CAT JSON entries - these already have all fields.
        """
        rows = []
        
        for entry in json_data:
            combined_cols = entry.get('combined_cols', [])
            combined_col_str = ' AND '.join(combined_cols) if combined_cols else 'N/A'
            
            gpt_output = entry.get('gpt_output', {})
            inferences_list = gpt_output.get('inferences', [])
            recommendations_list = gpt_output.get('final_product_recommendations', [])
            
            for idx, inference_obj in enumerate(inferences_list):
                recommended_product = ''
                if idx < len(recommendations_list):
                    recommended_product = recommendations_list[idx].get('title', '')
                
                overall_product = ''
                if recommendations_list:
                    overall_product = recommendations_list[0].get('title', '')
                
                row = {
                    'combined_col': combined_col_str,
                    'inference': inference_obj.get('inference', ''),
                    'inference_key': extract_keywords(inference_obj.get('inference', '')),
                    'category': inference_obj.get('category', ''),
                    'activity': inference_obj.get('activity', ''),
                    'reason': inference_obj.get('reason', ''),
                    'recommended_product': recommended_product,
                    'uncommonness': str(inference_obj.get('uncommonness', '')),
                    'sensitivity': str(inference_obj.get('sensitivity', '')),
                    'overall_recommended_product': overall_product
                }
                rows.append(row)
        
        return rows
    
    # Process both datasets
    print("Processing RAG data (deriving category/activity/reason from explanation)...")
    rag_rows = flatten_rag_entries(rag_data)
    
    print("Processing Cat data (using existing category/activity/reason)...")
    cat_rows = flatten_cat_entries(cat_data)
    
    # Combine all rows
    all_rows = rag_rows + cat_rows
    
    # CSV columns
    fieldnames = [
        'combined_col',
        'inference',
        'inference_key',
        'category',
        'activity',
        'reason',
        'recommended_product',
        'uncommonness',
        'sensitivity',
        'overall_recommended_product'
    ]
    
    # Write CSV
    print(f"Writing to {output_csv_path}...")
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(all_rows)
    
    # Summary
    print(f"\n✓ Conversion complete!")
    print(f"  Output file: {output_csv_path}")
    print(f"  Total rows: {len(all_rows)}")
    print(f"    - From RAG data: {len(rag_rows)}")
    print(f"    - From Cat data: {len(cat_rows)}")
    
    # Show sample
    if all_rows:
        print(f"\n  Sample row (RAG):")
        print(f"    Category: {rag_rows[0]['category']}")
        print(f"    Activity: {rag_rows[0]['activity']}")
        print(f"    Reason: {rag_rows[0]['reason'][:80]}...")


def main():
    """Main execution function"""
    RAG_JSON_PATH = '/scratch/cjpenni/departmental-honors/data_pipeline/inference_data/06OCT2025/new_inferences_rag.json'
    CAT_JSON_PATH = '/scratch/cjpenni/departmental-honors/data_pipeline/inferences_with_cat.json'
    OUTPUT_CSV_PATH = '/scratch/cjpenni/departmental-honors/data_pipeline/combined_inferences.csv'
    
    # Verify input files exist
    if not Path(RAG_JSON_PATH).exists():
        print(f"Error: {RAG_JSON_PATH} not found")
        return
    
    if not Path(CAT_JSON_PATH).exists():
        print(f"Error: {CAT_JSON_PATH} not found")
        return
    
    try:
        convert_json_to_csv_fixed(RAG_JSON_PATH, CAT_JSON_PATH, OUTPUT_CSV_PATH)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except IOError as e:
        print(f"Error: File I/O issue - {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
