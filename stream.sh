#!/bin/bash
#SBATCH --job-name=232-app
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4         # Use 4 CPU tasks (for data loading)
#SBATCH --mem=16G
#SBATCH --output=streamlit_output.log
#SBATCH --error=streamlit_error.log

# Optional: Print the node and port info for tunneling
echo "Job running on $(hostname)"
echo "To access the Streamlit app, use:"
echo "ssh -N -L 8501:localhost:8501 $USER@$(hostname)"

# Run the Streamlit app
streamlit run disaster_app.py --server.port 8501 --server.address 0.0.0.0
