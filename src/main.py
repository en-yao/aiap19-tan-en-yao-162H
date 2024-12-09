from model_evaluation import ModelEvaluation
import config as config

if __name__ == "__main__":
    pipeline = ModelEvaluation('config.py')
    pipeline.run_evaluation()