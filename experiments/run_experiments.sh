#!/bin/bash

echo "Running 01_XGB_vs_XGB_heart.py..."
python 01_XGB_vs_XGB_heart.py

if [ $? -ne 0 ]; then
    echo "Error: 01_XGB_vs_XGB_heart.py failed to execute."
    exit 1
fi

echo "Running 01_XGB_vs_XGB_cars.py..."
python 01_XGB_vs_XGB_cars.py

if [ $? -ne 0 ]; then
    echo "Error: 01_XGB_vs_XGB_cars.py failed to execute."
    exit 1
fi

echo "Running 01_XGB_vs_XGB_wine.py..."
python 01_XGB_vs_XGB_wine.py

if [ $? -ne 0 ]; then
    echo "Error: 01_XGB_vs_XGB_wine.py failed to execute."
    exit 1
fi

echo "Running 01_XGB_vs_XGB_MIMICIII_3_classes.py..."
python3 01_XGB_vs_XGB_MIMICIII_3_classes.py

if [ $? -ne 0 ]; then
    echo "Error: 01_XGB_vs_XGB_MIMICIII_3_classes.py failed to execute."
    exit 1
fi

echo "Running 01_XGB_vs_XGB_MIMICIII.py..."
python3 01_XGB_vs_XGB_MIMICIII.py

if [ $? -ne 0 ]; then
    echo "Error: 01_XGB_vs_XGB_MIMICIII.py failed to execute."
    exit 1
fi

echo "Running 02_XGB_vs_others_heart.py..."
python 02_XGB_vs_others_heart.py

if [ $? -ne 0 ]; then
    echo "Error: 02_XGB_vs_others_heart.py failed to execute."
    exit 1
fi

echo "Running 02_XGB_vs_others_cars.py..."
python 02_XGB_vs_others_cars.py

if [ $? -ne 0 ]; then
    echo "Error: 02_XGB_vs_others_cars.py failed to execute."
    exit 1
fi

echo "Running 02_XGB_vs_others_wine.py..."
python 02_XGB_vs_others_wine.py

if [ $? -ne 0 ]; then
    echo "Error: 02_XGB_vs_others_wine.py failed to execute."
    exit 1
fi

echo "Running 02_XGB_vs_others_MIMICIII_3_classes.py..."
python3 02_XGB_vs_others_MIMICIII_3_classes.py

if [ $? -ne 0 ]; then
    echo "Error: 02_XGB_vs_others_MIMICIII_3_classes.py failed to execute."
    exit 1
fi

echo "Running 02_XGB_vs_others_MIMICIII.py..."
python3 02_XGB_vs_others_MIMICIII.py

if [ $? -ne 0 ]; then
    echo "Error: 02_XGB_vs_others_MIMICIII.py failed to execute."
    exit 1
fi

echo "All Python scripts executed successfully!"