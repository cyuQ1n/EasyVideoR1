
# =====================================================================                                                                                                                                                                                                                                    
#  数据源:                                                                                                                                                 
#    - https://huggingface.co/datasets/MCG-NJU/ODV-Bench                                                                                                   
#    - https://huggingface.co/datasets/stdKonjac/LiveSports-3K                                                                                             
# =====================================================================     

cd EasyVideoR1/eval/preprocess

python cilp.py \
    --input_json ../data/valid_data/odvbench.json \
    --video_root /path/to/ODV-Bench \
    --output_dir ../data/ODV-Bench/clips \
    --format odvbench


python cilp.py \
    --input_json ../data/valid_data/livesports3k_qa.json \
    --video_root /path/to/LiveSports-3K/videos \
    --output_dir ../data/LiveSports-3K-QA/clips \
    --format livesports
