#!/bin/bash

if docker start postgres-latest-riskmaster; then
	echo "Postgres container started"
else
	echo "Failed to start Postgres container"
	exit 1
fi

echo "Starting Streamlit app..."
streamlit run app/app.py || echo "Failed to start Streamlit app"
