from model_config.model_config import build_training_config
from optimizer_utils import train_one_wnae_model, minimal_ptheta_profile_chisquare

class DummyTrial:
    def __init__(self, number=0):
        self.number = number
    def suggest_int(self, name, low, high):
        return low
    def suggest_float(self, name, low, high):
        return low

# ---------------- Build config ---------------- #
dataset_config_file = "../data/dataset_config_small.json" 
model_name = "feat16_encoder64_deep_qcd"
training_config = build_training_config(model_name, dataset_config_file)
training_config["N_EPOCHS"]=1
training_config["NUM_SAMPLES"]=2**18
#training_config["WNAE_PARAMS"]["initial_distribution"]="uniform"
training_config["WNAE_PARAMS"]["initial_distribution"]="gaussian"
training_config["WNAE_PARAMS"]["temperature"]=1.0
training_config["WNAE_PARAMS"]["step_size"]=None
training_config["WNAE_PARAMS"]["n_steps"]=10
training_config["WNAE_PARAMS"]["buffer_size"]=4096

# ---------------- Run one training ---------------- #
trial = DummyTrial()
summary, model = train_one_wnae_model(training_config, trial, model_number=1)

print("Training summary keys:", summary.keys())
print("Last training loss:", summary["training_losses"][-1])
print("Last validation loss:", summary["validation_losses"][-1])

from utils.jet_dataset import JetDataset
val_dataset = JetDataset(training_config["DATA_PATH"], indices=None, input_dim=training_config["INPUT_DIM"])
avg_chi2 = minimal_ptheta_profile_chisquare(model, val_dataset, n_samples=10000, n_bins=20, bounds=(-4,4), use_temperature=True,name="test_temp_gaussian_mcmc_sampling_data_eval")
#avg_chi2 = minimal_ptheta_profile_chisquare(model, val_dataset, n_samples=10000, n_bins=20, bounds=(-4,4), use_temperature=True,name="test_temp_uniform_mcmc_sampling_uniform_eval",evaluate_on_uniform=True)
#avg_chi2 = minimal_ptheta_profile_chisquare(model, val_dataset, n_samples=10000, n_bins=20, bounds=(-4,4), use_temperature=False,name="test_no_temp_gaussian_mcmc_sampling_data_eval")
#avg_chi2 = minimal_ptheta_profile_chisquare(model, val_dataset, n_samples=10000, n_bins=20, bounds=(-4,4), use_temperature=False,name="test_no_temp_uniform_mcmc_sampling_uniform_eval",evaluate_on_uniform=True)
print("Average chi2:", avg_chi2)
