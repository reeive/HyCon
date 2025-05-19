import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import torch 
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from src.models.feature_extractors import ResNet50FeatureExtractor
from src.main_experiment import exp 

@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main_app(cfg: DictConfig) -> None:
    logging.info("Loaded configuration:")
    logging.info(OmegaConf.to_yaml(cfg))

    resnet_extractor = ResNet50FeatureExtractor()
    
    results = exp(cfg, resnet_extractor)
    
    try:
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        pseudo_labels_path = os.path.join(output_dir, "student_soft_pseudo_labels.pt")
        torch.save(results['student_soft_pseudo_labels'], pseudo_labels_path)
        logging.info(f"Saved student soft pseudo-labels to: {pseudo_labels_path}")
    except Exception as e:
        logging.error(f"Error saving pseudo-labels: {e}")
        # Fallback save location if hydra output_dir is not available (e.g. during debugging)
        fallback_path = "student_soft_pseudo_labels_fallback.pt"
        torch.save(results['student_soft_pseudo_labels'], fallback_path)
        logging.info(f"Saved student soft pseudo-labels to fallback: {fallback_path}")


    logging.info("Experiment finished.")

if __name__ == "__main__":
    main_app()