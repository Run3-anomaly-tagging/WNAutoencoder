import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from wnae import WNAE
from utils.jet_dataset import JetDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pprint

DEVICE = torch.device("cpu")

def run_training(model, optimizer, n_epochs, training_loader, validation_loader):

    training_losses = []
    validation_losses = []

    for i_epoch in range(n_epochs):
        model.train()
        training_loss = 0
        n_batches = 0

        bar_format = f"Epoch {i_epoch+1}/{n_epochs}: " + "{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"

        for batch in tqdm(training_loader, bar_format=bar_format):
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            loss, train_dict = model.train_step(x)
            loss.backward()
            optimizer.step()
            training_loss += train_dict["loss"]
            n_batches += 1

            print(f"Positive energy: {train_dict['positive_energy']:.4f}, Negative energy: {train_dict['negative_energy']:.4f}")

        training_losses.append(training_loss / n_batches)

        # Validation
        model.eval()
        validation_loss = 0
        n_batches = 0

        for batch in validation_loader:
            x = batch[0].to(DEVICE)
            val_dict = model.validation_step(x)
            validation_loss += val_dict["loss"]
            n_batches += 1

        validation_losses.append(validation_loss / n_batches)

        print(f"Epoch {i_epoch+1}/{n_epochs} | Train Loss: {training_losses[-1]:.4f} | Val Loss: {validation_losses[-1]:.4f}")
    return training_losses, validation_losses

def train_one_wnae_model(training_config, trial, model_number=None):
    """
    Train one WNAE model with trial-suggested hyperparameters.
    Returns a 'summary' dict compatible with your chi2 metric.
    """

    # ----------------- Update hyperparameters from trial -----------------
    #n_steps = trial.suggest_int("n_steps", 10, 50)
    #step_size = trial.suggest_float("step_size", 0.05, 0.5)

    wnae_params = training_config["WNAE_PARAMS"].copy()
    #wnae_params.update({"n_steps": n_steps, "step_size": step_size})
    learning_rate = training_config["LEARNING_RATE"]
    # ---------------------------------------------------------------------

    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    dataset = JetDataset(training_config["DATA_PATH"])
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    split = int(0.8 * len(indices))
    train_dataset = JetDataset(training_config["DATA_PATH"], indices=indices[:split], input_dim=training_config["INPUT_DIM"])
    val_dataset = JetDataset(training_config["DATA_PATH"], indices=indices[split:], input_dim=training_config["INPUT_DIM"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["BATCH_SIZE"],
        sampler=RandomSampler(train_dataset, replacement=True, num_samples=training_config["NUM_SAMPLES"])
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["BATCH_SIZE"],
        sampler=RandomSampler(val_dataset, replacement=True, num_samples=training_config["NUM_SAMPLES"])
    )
    print("WNAE params")
    print(wnae_params)
    model_config = training_config["MODEL_CONFIG"]
    model = WNAE(
        encoder=model_config["encoder"](),
        decoder=model_config["decoder"](),
        **wnae_params
    ).to(DEVICE)
    pprint.pprint(vars(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


    training_losses, validation_losses = run_training(model,optimizer,training_config["N_EPOCHS"],train_loader,val_loader)

    summary = {
        "training_name": f"trial_{trial.number}_model_{model_number}",
        "training_losses": training_losses,
        "validation_losses": validation_losses,
    }

    return summary, model

def minimal_ptheta_profile_chisquare(model, val_dataset, n_samples=10000, n_bins=50, bounds=(-4,4), use_temperature=False,name="test",evaluate_on_uniform=False):
    """
    Compute average chi2 between validation data (used to compute p_theta) 
    and MCMC-generated samples using model probabilities from energies.

    Args:
        model: trained WNAE model
        val_dataset: Dataset with validation samples
        n_samples: number of validation and MCMC samples
        n_bins: number of histogram bins per feature
        bounds: histogram bounds

    Returns:
        avg_chi2: scalar chi2 averaged over all features
    """
    import numpy as np
    import torch

    model.eval()

    val_data = torch.stack([val_dataset[i][0] for i in range(n_samples)]).to(DEVICE)
    input_dim = val_data.shape[1]
    if use_temperature:
        temperature = model.temperature
        print(f"Using model temperature for scaling energy: T={temperature:.3f}")
    else:
        temperature = 1.0

    #Sanity check to see if the buffer is empty (shouldn't be!)
    print("Buffer length before MCMC: ", len(model.buffer.buffer))

    negative_samples = model.run_mcmc(n_samples=n_samples, replay=True).detach().cpu().numpy()

    chi2_list = []
    low, high = bounds
    bin_edges = np.linspace(low, high, n_bins+1)
    bin_width = bin_edges[1] - bin_edges[0]

    all_hist_data  = []
    all_energies  = []
    all_hist_neg  = []
    all_p_theta  = []

    uniform_samples = torch.empty((n_samples, input_dim), device=DEVICE).uniform_(low, high)
    with torch.no_grad():
        if evaluate_on_uniform:
            eval_dict = model.evaluate(uniform_samples)
            energies_all = eval_dict["reco_errors"].cpu().numpy()
            feature_vals_energy = uniform_samples.cpu().numpy()
        else:
            eval_dict = model.evaluate(val_data)
            energies_all = eval_dict["reco_errors"].cpu().numpy()
            feature_vals_energy = val_data.cpu().numpy()

    for i in range(input_dim):
        feature_vals_data = val_data[:, i].cpu().numpy()
        feature_vals_for_energy = feature_vals_energy[:, i]  #either data or uniform random distribution
        energies = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        counts, _ = np.histogram(feature_vals_for_energy, bins=bin_edges)
        sums, _ = np.histogram(feature_vals_for_energy, bins=bin_edges, weights=energies_all)
        energies = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts>0)        

        all_energies.append(energies)
        q_theta = np.exp(-energies/temperature)
        q_sum = np.sum(q_theta)
        p_theta = q_theta / q_sum if q_sum > 0 else np.zeros_like(q_theta)
        p_theta = p_theta / bin_width
        # Histogram negative samples for this feature
        neg_feature = negative_samples[:, i]
        hist_neg, _ = np.histogram(neg_feature, bins=bin_edges, density=True)
        hist_data, _ = np.histogram(feature_vals_data, bins=bin_edges, density=True)

        all_hist_data.append(hist_data)
        all_hist_neg.append(hist_neg)
        all_p_theta.append(p_theta)


        mask = p_theta > 0
        #chi2 = np.sum((hist_neg[mask] - p_theta[mask])**2 / (p_theta[mask] + 1e-8))
        chi2 = 0.5 * np.sum((hist_neg[mask] - p_theta[mask])**2 / (hist_neg[mask] + p_theta[mask] + 1e-8))#Not really chi2, but the original chi2 penalizes strongly when p_theta is narrower than p_MCMC, this approach is more symmetric
        chi2_list.append(chi2)

    avg_chi2 = np.mean(chi2_list)

    if val_dataset[0][0].shape[0] == 2:
        print("Input dimension is 2 — plotting 2D energy map for sanity check.")
        plot_energy_2d_map(model, bounds=bounds, n_points=100, output_path=f"{name}_energy_map.pdf")

    plot_output_path = f"{name}.pdf"
    plot_feature_histograms_ptheta(np.array(all_hist_data),np.array(all_hist_neg),np.array(all_p_theta),bin_edges,chi2_list, avg_chi2,output_path=plot_output_path, energies=np.array(all_energies))
    #plot_feature_histograms_ptheta(np.array(all_hist_data),np.array(all_hist_neg),np.array(all_p_theta), np.array(all_energies),bin_edges,chi2_list, avg_chi2,output_path="feature_histograms.pdf")

    return avg_chi2

def plot_feature_histograms_ptheta(hist_data, hist_neg, p_theta, bin_edges, chi2_all, avg_chi2, output_path, energies=None):
    """
    Plot p_data, p_neg (MCMC), and p_theta on the same figure for each feature.

    Args:
        hist_data: 2D array, shape [n_features, n_bins] of normalized histograms for data
        hist_neg: 2D array, shape [n_features, n_bins] of normalized histograms for negative samples
        p_theta: 2D array, shape [n_features, n_bins] of model probabilities
        bin_edges: 1D array of bin edges used for histograms
        avg_chi2: scalar average chi2 (for filename or title)
        output_path: path to save the PDF
    """
    n_features = hist_data.shape[0]
    with PdfPages(output_path) as pdf:
        for i in range(n_features):
            plt.figure()
            fig, ax1 = plt.subplots()
            ax1.step(bin_edges, np.append(hist_data[i], hist_data[i][-1]), where='post', label=r'$p_{data}$', linewidth=1.5)#np.append(hist_data[i], hist_data[i][-1]) is just a visual trick to plot histogram to last bin, no effect on the plot content
            ax1.step(bin_edges, np.append(hist_neg[i], hist_neg[i][-1]), where='post', label=r'$p_{MCMC}$', linewidth=1.5)
            ax1.step(bin_edges, np.append(p_theta[i], p_theta[i][-1]), where='post', label=r'$p_{\theta}$', linewidth=1.5)
            ax1.set_xlabel(f"Feature {i}")
            ax1.set_ylabel("Probability density")
            ax1.margins(y=0.35)
            ax1.set_ylim(bottom=0)
            
            if energies is not None:
                ax2 = ax1.twinx()
                ax2.set_ylim(0,np.max(energies[i])*1.5)
                ax2.step(bin_edges, np.append(energies[i], energies[i][-1]), where='post', label='Energy', linewidth=1.5, linestyle='--')
                ax2.set_ylabel("Reco. energy")
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False)
            else:
                ax1.legend(frameon=False)

            plt.title(f"Feature {i}, $\\chi^2$ = {chi2_all[i]:.2f}, avg. $\\chi^2$ = {avg_chi2:.2f}")
            plt.xlim(bin_edges[0], bin_edges[-1])
            pdf.savefig()
            plt.close()

def plot_energy_2d_map(model, bounds=(-4, 4), n_points=100, output_path="energy_map_2d.pdf"):
    """
    Plots the 2D energy landscape (reconstruction error) of the model
    evaluated on a uniform grid over the given bounds.
    Only makes sense for 2D inputs (useful for debugging).

    Args:
        model: trained WNAE model
        bounds: (low, high) range for each axis
        n_points: number of grid points per dimension
        output_path: where to save the PDF
    """

    low, high = bounds
    x = np.linspace(low, high, n_points)
    y = np.linspace(low, high, n_points)
    X, Y = np.meshgrid(x, y)

    # Build uniform grid [n_points^2, 2]
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)
    grid_tensor = torch.tensor(grid, dtype=torch.float32, device=next(model.parameters()).device)

    # Evaluate model energy (reco error)
    model.eval()
    with torch.no_grad():
        eval_dict = model.evaluate(grid_tensor)
        energies = eval_dict["reco_errors"].cpu().numpy()

    # Reshape back to 2D map
    E = energies.reshape(n_points, n_points)

    # Plot
    with PdfPages(output_path) as pdf:
        plt.figure(figsize=(6, 5))
        plt.title("2D Energy Landscape")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.pcolormesh(X, Y, E, shading="auto")
        plt.colorbar(label="Reconstruction error")
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    print(f"Saved 2D energy map to {output_path}")