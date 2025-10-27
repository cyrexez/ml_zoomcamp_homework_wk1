FROM agrigorev/zoomcamp-model:2025

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies using uv
RUN uv sync --frozen

# Copy the pipeline and application files
COPY pipeline_v1.bin .
COPY webpredict.py .
COPY webtest.py .

# Expose the port
EXPOSE 9696

# Run the application with uvicorn
CMD ["uv", "run", "uvicorn", "webpredict:app", "--host", "0.0.0.0", "--port", "9696"]