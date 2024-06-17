FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Install required libraries
# RUN pip install google-cloud-storage pandas pyarrow scikit-learn
RUN pip install pandas pyarrow scikit-learn==

# Copy your script and the service account key into the container
COPY your_script.py /app/your_script.py
COPY service-account-key.json /app/service-account-key.json

# Set environment variable for authentication
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service-account-key.json"

# Set the working directory
WORKDIR /app

# Command to run your script
CMD ["python", "your_script.py"]
