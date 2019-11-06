#!/bin/sh

if [ "$1" = "process_data" ]; then 
    python3 Composer/load_songs.py --load_path=data/raw_data/maestro-v2.0.0/maestro-v2.0.0.csv --save_path=data/processed_data
else
    echo "Invalid option"
fi 